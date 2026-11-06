"""Guild purchases must honor remaining limits, currency, quantities, and prices."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app import Clicker
from digit_ocr import DigitReader, NumberReading, CounterReading
from purchase_checks import currency_floor, stock_check, quantity_check, QuantityTemplates, confirmation_check, read_stock_counter
from drift.tasks.features import FEATURE_BY_ID
from dev_tools.configure_guild_shop import presets


class GuildShopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=json.loads((ROOT/'config.json').read_text(encoding='utf-8-sig'))
        cls.rules={r['id']:r for r in cls.config['rules'] if r['id'] in FEATURE_BY_ID['guild_shop']['rules']}
        cls.reader=DigitReader(ROOT)
        cls.frames={p.stem:np.asarray(Image.open(p).convert('RGB')) for p in (ROOT/'assets'/'guild-shop').glob('*.png')}
        cls.context=SimpleNamespace(root=ROOT,config=cls.config)

    def match(self,name,frame):
        rule,button,offset=Clicker.load_rule(self.context,self.rules['driftinggirl_guild_'+name])
        return button.match(frame,offset=offset,similarity=rule['threshold']),button

    def test_balance_and_stock_reject_unclear_and_cap_affordability(self):
        read=lambda text,confidence=.99:NumberReading(text,int(text) if text.isdigit() else None,confidence)
        self.assertEqual(currency_floor(read('4.22k'),.93),4210)
        for text in ('','4.2m','-10','10/20','4k?'):
            self.assertIsNone(currency_floor(read(text),.93))
        condition=self.rules['driftinggirl_guild_potion_select']['purchase_condition']
        for used,balance,target in ((0,'100',10),(4,'100',6),(4,'25',2),(10,'100',0),(0,'9',0)):
            passed,_,result=stock_check(CounterReading(f'{used}/10',used,10,.99),read(balance),condition)
            self.assertEqual(result['target'],target)
            self.assertEqual(passed,target>0)
            skipped=stock_check(CounterReading('',used,10,.99),read(balance),{**condition,'mode':'unavailable'})
            self.assertEqual(skipped[0],target==0)
        for counter,balance in ((CounterReading('',None,None,0),read('100')),
                (CounterReading('0/5',0,5,.99),read('100')),(CounterReading('0/10',0,10,.9),read('100')),
                (CounterReading('0/10',0,10,.99),read('100',.9))):
            for mode in ('available','unavailable'):
                self.assertFalse(stock_check(counter,balance,{**condition,'mode':mode})[0])

    def test_quantity_must_reach_target_and_price_must_equal_bundles(self):
        stock={'valid':True,'target':5,'unit_price':20,'balance':100}
        c=self.rules['driftinggirl_guild_material_buy']['quantity_condition']
        r=lambda value,confidence=.99:NumberReading(str(value),value,confidence)
        self.assertTrue(quantity_check(r(250),r(100),stock,c)[0])
        self.assertTrue(quantity_check(r(200),r(80),stock,{**c,'mode':'increment'})[0])
        for q,p in ((200,80),(250,20),(300,120),(251,100),(50,100),(None,100)):
            self.assertFalse(quantity_check(r(q),r(p),stock,c)[0])
        self.assertFalse(quantity_check(r(250,.9),r(100),stock,c)[0])
        self.assertTrue(quantity_check(r(None),r(40),stock,{**c,'mode':'increment'})[0])
        self.assertFalse(quantity_check(r(None),r(100),stock,c)[0])
        self.assertFalse(quantity_check(r(150),r(40),stock,{**c,'mode':'increment'})[0])
        self.assertFalse(quantity_check(r(250),r(100),None,c)[0])
        self.assertFalse(quantity_check(r(250),r(100),{**stock,'balance':99},c)[0])
        self.assertTrue(quantity_check(r(100),r(40),{**stock,'target':2,'balance':45},c)[0])

    def test_daily_completion_requires_both_actual_stock_limits_not_low_balance(self):
        context=SimpleNamespace(root=ROOT,lock=threading.RLock(),digit_reader=self.reader,
                                state={'frames':1,'numbers':{}})
        context.purchase_condition=lambda rule,button,frame:Clicker.purchase_condition(context,rule,button,frame)
        matchers={key:Clicker.load_rule(self.context,rule) for key,rule in self.rules.items()}
        close=self.rules['driftinggirl_guild_close']
        for name,expected in (('guild',False),('guild-potion-bought',False),('guild-all-bought',True)):
            with self.subTest(frame=name):
                self.assertEqual(Clicker.daily_completion_visible(context,'guild_shop',close,self.frames[name],matchers),expected)
        def insufficient(rule,button,frame):
            context.state['numbers'][rule['id']]={'valid':True,'remaining':3,'target':0,'balance':0}
            return True,None
        context.purchase_condition=insufficient
        self.assertFalse(Clicker.daily_completion_visible(context,'guild_shop',close,self.frames['guild-all-bought'],matchers))

    def test_shop_navigation_and_product_identity(self):
        self.assertTrue(self.match('open',self.frames['main'])[0])
        for name in ('open','special','guild'):
            self.assertFalse(self.match('open',self.frames[name])[0])
            self.assertTrue(self.match('special',self.frames[name])[0])
        for name in ('special','guild'):
            self.assertTrue(self.match('tab',self.frames[name])[0])
        for item in ('potion','material'):
            self.assertTrue(self.match(item+'_select',self.frames['guild'])[0])
            self.assertFalse(self.match(item+'_select',self.frames['special'])[0])
            self.assertFalse(self.match(item+'_buy',self.frames['guild'])[0])
            other='material' if item=='potion' else 'potion'
            self.assertFalse(self.match(item+'_buy',self.frames[other+'-min'])[0])
        self.assertFalse(self.match('close',self.frames['material-min'])[0])
        self.assertTrue(self.match('close',self.frames['guild'])[0])

    def test_native_quantity_and_price_at_every_increment(self):
        context=SimpleNamespace(root=ROOT,lock=threading.RLock(),digit_reader=self.reader,state={'frames':1,'numbers':{}})
        for item,maximum,bundle,unit in (('potion',10,1,10),('material',5,50,20)):
            stock_id='driftinggirl_guild_'+item+'_select'
            context.state['numbers'][stock_id]={'valid':True,'target':maximum,'unit_price':unit,'balance':1000}
            for count in range(1,maximum+1):
                frame=self.frames[item+'-%03d'%(count*bundle)]
                for suffix in ('plus','buy'):
                    with self.subTest(item=item,count=count,action=suffix):
                        rule=self.rules['driftinggirl_guild_'+item+'_'+suffix]
                        matched,button=self.match(item+'_'+suffix,frame)
                        self.assertTrue(matched)
                        passed,key=Clicker.purchase_condition(context,rule,button,frame)
                        self.assertEqual(key,(count*bundle,count*unit,maximum*bundle))
                        self.assertEqual(passed,(count<maximum if suffix=='plus' else count==maximum))
            bad=frame.copy();bad[838:920,286:382]=0
            self.assertFalse(self.match(item+'_buy',bad)[0])

    def test_sunrise_shop_entry_excludes_sky_inside_cart_handle(self):
        frame=np.asarray(Image.open(ROOT/'tests/fixtures/guild-selection/main-sunrise-20261030.png').convert('RGB'))
        for name in ('open','return'):
            saved=deepcopy(self.rules['driftinggirl_guild_'+name])
            rule,b,o=Clicker.load_rule(self.context,saved)
            self.assertTrue(b.match(frame,offset=o,similarity=rule['threshold']))
            self.assertGreaterEqual(b.last_similarity,.97)
            self.assertEqual(rule,saved)
            self.assertEqual(rule['threshold'],.94)
            if name=='open':self.assertEqual(tuple(b.button),(648,370,707,425))
            missing=frame.copy();missing[378:410,658:693]=0
            for negative in (missing,frame//2,self.frames['guild'],self.frames['material-confirm']):
                self.assertFalse(b.match(negative,offset=o,similarity=rule['threshold']))
            shifted=frame.copy();shifted[250:450,610:720]=0
            shifted[250:360,610:720]=frame[340:450,610:720]
            self.assertTrue(b.match(shifted,offset=o,similarity=rule['threshold']))
            if name=='open':self.assertEqual(tuple(b.button),(648,280,707,335))

    def test_invalid_purchase_link_and_numeric_modes(self):
        base=self.rules['driftinggirl_guild_material_buy']
        for change in ({'stock_rule':'missing'},{'unit_price':10},{'maximum':10},{'mode':'always'},
                       {'bundle_size':0},{'preprocess':'unknown'},{'unit_price':True}):
            rule=deepcopy(base);rule['quantity_condition'].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):
                Clicker.load_rule(self.context,rule)

    def test_stylized_five_and_seven_are_verified_even_as_final_targets(self):
        templates=QuantityTemplates(ROOT,'potion')
        for q in (5,7):
            image=self.frames['potion-%03d'%q][655:724,272:450]
            reading=templates.read(image)
            self.assertEqual(reading.value,q)
            self.assertGreaterEqual(reading.confidence,.985)
            c=self.rules['driftinggirl_guild_potion_buy']['quantity_condition']
            stock={'valid':True,'target':q,'unit_price':10,'balance':q*10}
            self.assertTrue(quantity_check(reading,NumberReading(str(q*10),q*10,.99),stock,c)[0])
        self.assertIsNone(templates.read(np.full((69,178,3),240,dtype=np.uint8)).value)

    def test_confirmation_amount_must_match_verified_current_purchase(self):
        selection=dict(kind='purchase_quantity',valid=True,passed=True,value=10,limit=10,price=100)
        context=SimpleNamespace(root=ROOT,lock=threading.RLock(),digit_reader=self.reader,
                                state={'frames':1,'numbers':{'driftinggirl_guild_potion_buy':selection}})
        rule=self.rules['driftinggirl_guild_potion_confirm']
        matched,button=self.match('potion_confirm',self.frames['potion-confirm'])
        self.assertTrue(matched)
        self.assertEqual(Clicker.purchase_condition(context,rule,button,self.frames['potion-confirm']),(True,(100,100)))
        condition=rule['confirmation_condition']
        for previous in (None,{**selection,'passed':False},{**selection,'valid':False},
                         {**selection,'value':9},{**selection,'price':20}):
            self.assertFalse(confirmation_check(NumberReading('100',100,.99),previous,condition)[0])
        for reading in (NumberReading('100',100,.90),NumberReading('20',20,.99),NumberReading('',None,0)):
            self.assertFalse(confirmation_check(reading,selection,condition)[0])
        for source in ('missing','driftinggirl_guild_potion_plus','driftinggirl_guild_material_buy'):
            bad=deepcopy(rule);bad['confirmation_condition']['quantity_rule']=source
            with self.assertRaises(ValueError):Clicker.load_rule(self.context,bad)
        wrong_currency=self.frames['potion-confirm'].copy();wrong_currency[556:621,211:275]=220
        matched,button=self.match('potion_confirm',wrong_currency)
        self.assertTrue(matched)  # title/button alone cannot authorize buying
        self.assertFalse(Clicker.purchase_condition(context,rule,button,wrong_currency)[0])
        self.assertFalse(self.match('potion_confirm',self.frames['potion-010'])[0])
        self.assertFalse(self.match('potion_confirm',self.frames['material-confirm'])[0])
        self.assertFalse(self.match('material_confirm',self.frames['potion-confirm'])[0])

    def test_centered_question_shift_keeps_currency_and_amount_guards(self):
        original=deepcopy(self.rules['driftinggirl_guild_potion_confirm'])
        loaded,_,_=Clicker.load_rule(self.context,original)
        self.assertEqual(loaded['required_source_areas'],[[309,410,410,438]])
        self.assertEqual(original,self.rules['driftinggirl_guild_potion_confirm'])
        selection=dict(kind='purchase_quantity',valid=True,passed=True,value=10,limit=10,price=100)
        context=SimpleNamespace(root=ROOT,lock=threading.RLock(),digit_reader=self.reader,
                                state={'frames':1,'numbers':{'driftinggirl_guild_potion_buy':selection}})
        for shift in (-25,25):
            frame=self.frames['potion-confirm'].copy()
            frame[550:625,180:540]=np.roll(frame[550:625,180:540],shift,axis=1)
            matched,button=self.match('potion_confirm',frame)
            self.assertTrue(matched)
            self.assertEqual(Clicker.purchase_condition(context,loaded,button,frame),(True,(100,100)))
            self.assertFalse(self.match('potion_confirm',frame//2)[0])

    def test_corrected_real_material_confirmation_is_80_not_100(self):
        rule=self.rules['driftinggirl_guild_material_confirm']
        matched,button=self.match('material_confirm',self.frames['material-confirm'])
        self.assertTrue(matched)
        selection=dict(kind='purchase_quantity',valid=True,passed=True,value=200,limit=200,price=80)
        context=SimpleNamespace(root=ROOT,lock=threading.RLock(),digit_reader=self.reader,
                                state={'frames':1,'numbers':{'driftinggirl_guild_material_buy':selection}})
        self.assertTrue(Clicker.purchase_condition(context,rule,button,self.frames['material-confirm'])[0])
        context.state['numbers']['driftinggirl_guild_material_buy']={**selection,'price':100}
        self.assertFalse(Clicker.purchase_condition(context,rule,button,self.frames['material-confirm'])[0])

    def test_actual_100_material_fault_normalizes_saved_rule_without_mutating_it(self):
        frame=np.asarray(Image.open(ROOT/'tests/fixtures/guild-selection/material-confirm-100-20261030.png').convert('RGB'))
        saved=deepcopy(self.rules['driftinggirl_guild_material_confirm'])
        saved['required_source_areas']=[[228,573,257,603],[338,576,528,605],[284,410,438,438]]
        original=deepcopy(saved)
        rule,button,offset=Clicker.load_rule(self.context,saved)
        self.assertEqual(saved,original)
        self.assertEqual(rule['required_source_areas'],[[284,410,438,438]])
        self.assertEqual(rule['threshold'],.94)
        self.assertTrue(button.match(frame,offset=offset,similarity=rule['threshold']))
        self.assertEqual(tuple(button.button),(111,752,331,822))
        selection=dict(kind='purchase_quantity',valid=True,passed=True,value=250,limit=250,price=100)
        context=SimpleNamespace(root=ROOT,lock=threading.RLock(),digit_reader=self.reader,
                                state={'frames':1,'numbers':{'driftinggirl_guild_material_buy':selection}})
        self.assertEqual(Clicker.purchase_condition(context,rule,button,frame),(True,(100,100)))
        for previous in (None,{**selection,'price':80},{**selection,'passed':False},{**selection,'value':200}):
            context.state['numbers']['driftinggirl_guild_material_buy']=previous
            self.assertFalse(Clicker.purchase_condition(context,rule,button,frame)[0])
        context.state['numbers']['driftinggirl_guild_material_buy']=selection
        missing_coin=frame.copy();missing_coin[556:621,211:275]=220
        self.assertTrue(button.match(missing_coin,offset=offset,similarity=rule['threshold']))
        self.assertFalse(Clicker.purchase_condition(context,rule,button,missing_coin)[0])
        for box in ((284,410,438,438),(130,754,298,820)):
            rejected=frame.copy();x,y,X,Y=box;rejected[y:Y,x:X]=0
            self.assertFalse(button.match(rejected,offset=offset,similarity=rule['threshold']))
        self.assertFalse(button.match(frame//2,offset=offset,similarity=rule['threshold']))
        custom,custom_button,custom_offset=Clicker.load_rule(self.context,{**saved,'id':'custom_material_confirm'})
        self.assertEqual(custom['required_source_areas'],saved['required_source_areas'])
        self.assertFalse(custom_button.match(frame,offset=custom_offset,similarity=custom['threshold']))

    def test_material_centered_price_is_fresh_and_currency_guarded_for_two_and_three_digits(self):
        frames=[(self.frames['material-confirm'],200,80),
                (np.asarray(Image.open(ROOT/'tests/fixtures/guild-selection/material-confirm-100-20261030.png').convert('RGB')),250,100)]
        for frame,quantity,price in frames:
            for shift in (-25,25):
                moved=frame.copy();moved[550:625,180:540]=np.roll(moved[550:625,180:540],shift,axis=1)
                rule,b,o=Clicker.load_rule(self.context,self.rules['driftinggirl_guild_material_confirm'])
                self.assertTrue(b.match(moved,offset=o,similarity=rule['threshold']))
                selection=dict(kind='purchase_quantity',valid=True,passed=True,value=quantity,limit=quantity,price=price)
                ctx=SimpleNamespace(root=ROOT,lock=threading.RLock(),digit_reader=self.reader,
                    state={'frames':1,'numbers':{'driftinggirl_guild_material_buy':selection}})
                self.assertEqual(Clicker.purchase_condition(ctx,rule,b,moved),(True,(price,price)))

    def test_native_sold_out_potion_is_skipped_and_material_remains_available(self):
        context=SimpleNamespace(root=ROOT,lock=threading.RLock(),digit_reader=self.reader,state={'frames':1,'numbers':{}})
        frame=self.frames['guild-potion-bought']
        for name,expected in (('potion_select',False),('potion_skip',True),('material_select',True)):
            rule=self.rules['driftinggirl_guild_'+name]
            matched,button=self.match(name,frame)
            if name!='potion_select':self.assertTrue(matched)
            self.assertEqual(Clicker.purchase_condition(context,rule,button,frame)[0],expected)
        frame=self.frames['guild-all-bought']
        for name in ('potion_skip','material_skip'):
            rule=self.rules['driftinggirl_guild_'+name]
            matched,button=self.match(name,frame)
            self.assertTrue(matched)
            self.assertTrue(Clicker.purchase_condition(context,rule,button,frame)[0])
        self.assertIsNone(read_stock_counter(self.reader,np.full((28,68,3),90,dtype=np.uint8)).current)

    def test_complete_replay_adds_to_both_limits_and_exits(self):
        sequence=[('open','main'),('special','open'),('tab','special'),('potion_select','guild')]
        sequence += [('potion_plus','potion-%03d'%q) for q in range(1,10)]
        sequence += [('potion_buy','potion-010'),('potion_confirm','potion-confirm'),('material_select','guild-potion-bought')]
        sequence += [('material_plus','material-%03d'%q) for q in range(50,250,50)]
        sequence += [('material_buy','material-250'),('material_confirm','material-confirm'),('close','guild-all-bought')]
        frames=dict(self.frames)
        # Synthetic replay only: reuse the real 100-price question for the
        # simulated five-bundle purchase. It is not real-world purchase proof.
        frames['material-confirm']=frames['material-confirm'].copy()
        frames['material-confirm'][550:625,180:540]=frames['potion-confirm'][550:625,180:540]
        class ReplayDevice:
            def __init__(self,config,stop):self.stop,self.image,self.taps=stop,None,[]
            def screenshot(self):
                self.image=frames[sequence[len(self.taps)][1] if len(self.taps)<len(sequence) else 'main'].copy()
                return self.image
            def click(self,button):
                if self.stop.is_set():raise InterruptedError()
                index=len(self.taps)
                if index>=len(sequence) or button.name!='driftinggirl_guild_'+sequence[index][0]:
                    raise AssertionError('Unexpected purchase: '+button.name)
                self.taps.append(button.name)
            def stuck_record_add(self,button):pass
        with tempfile.TemporaryDirectory(prefix='guild-replay-') as directory:
            root=Path(directory);shutil.copytree(ROOT/'assets'/'guild-shop',root/'assets'/'guild-shop')
            config=deepcopy(self.config);config['poll_interval']=.1;config['rules']=deepcopy(list(self.rules.values()))
            for r in config['rules']:r.update(enabled=True,interval=.2,next_delay=0,stage_timeout=5)
            (root/'config.json').write_text(json.dumps(config),encoding='utf-8')
            app=Clicker(root,ReplayDevice);app.digit_reader=self.reader
            try:
                app.start(preview=False,stop_on_main=True,max_clicks=30,timeout=30)
                app.thread.join(33)
                self.assertFalse(app.running());self.assertIsNone(app.state['last_error'])
                self.assertEqual(app.state['stop_reason'],'cycle_complete')
                self.assertEqual(app.state['clicks'],23)
                self.assertTrue(app.daily_schedule.is_complete('guild_shop'))
            finally:
                app.stop()
                if app.thread:app.thread.join(3)
                for handler in list(app.log.handlers):handler.close();app.log.removeHandler(handler)


if __name__=='__main__':unittest.main()
