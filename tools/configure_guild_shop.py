"""Install the native-resolution guild shop preset without changing other tasks."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'driftinggirl_guild_'


def rule(key, name, asset, crop, search, stage, following, click=None, **extra):
    result = dict(id=PREFIX+key, name='公会商品：'+name, enabled=True,
                  asset='assets/guild-shop/'+asset+'.png', source_crop=crop,
                  search_area=search, scales=[1.0], threshold=.94, interval=1,
                  stable_frames=3, stage=stage, next_stage=following,
                  next_delay=.6, stage_timeout=30)
    if click is not None:
        result['click_source_area'] = click
    result.update(extra)
    return result


def presets():
    bright = dict(source_area=[659,378,691,406], hsv=[[0,0,140],[179,255,255]], minimum_fraction=.4)
    header = [175,337,328,374]
    page_parts = [[91,469,132,511],[526,350,553,376]]
    rules = [
        rule('open','打开商店','main',[649,371,698,415],[610,340,720,450],
             'main','guild_special',[648,370,707,425],interval=3600,color_condition=bright),
        rule('special','进入 SPECIAL 页','open',[457,255,552,283],[420,220,590,310],
             'guild_special','guild_tab',[450,232,562,296],required_source_areas=[[637,251,688,294]]),
        rule('tab','切换公会点数页','special',[89,466,136,514],[64,433,157,546],
             'guild_tab','guild_potion',[77,448,146,533],required_source_areas=[[457,255,552,283]]),
    ]
    for item,label,cap,unit,bundle,icon,counter,unit_parts,title,product,following in (
        ('potion','药水',10,10,1,[522,406,594,465],[522,502,593,530],
         [[505,545,535,575],[568,548,598,574]],[309,410,410,455],[305,490,424,641],'guild_material'),
        ('material','天赋值',5,20,50,[365,407,447,461],[373,502,441,530],
         [[354,545,383,575],[419,548,449,574]],[284,410,438,455],[288,490,430,641],'guild_close'),
    ):
        stock = dict(label=label+'已购份数', counter_area=counter, balance_area=[560,350,641,377],
                     expected_total=cap, unit_price=unit, confidence=.93, mode='available')
        common = dict(required_source_areas=page_parts+unit_parts)
        select = rule(item+'_select','选择'+label,'guild',header,[150,312,355,404],
                      'guild_'+item,'guild_'+item+'_quantity',icon,purchase_condition=stock,**common)
        select['required_source_areas'] = common['required_source_areas']+[icon]
        skip = rule(item+'_skip',label+'已买满或点数不足时跳过','guild',header,[150,312,355,404],
                    'guild_'+item,following,action='observe',purchase_condition={**stock,'mode':'unavailable'},**common)
        quantity = dict(label=label+'本次购买数量',stock_rule=select['id'],maximum=cap,
                        unit_price=unit,bundle_size=bundle,quantity_area=[272,655,450,724],
                        price_area=[275,846,529,905],preprocess='tight' if item=='potion' else 'purple_text',
                        template_prefix=item,confidence=.93)
        parts = [product,[323,852,359,894],[500,741,536,767]]
        plus = rule(item+'_plus',label+'逐次加一','%s-min'%item,title,[250,384,475,480],
                    'guild_'+item+'_quantity','guild_'+item+'_quantity',[487,739,549,768],
                    required_source_areas=parts,required_tolerance=30,stable_frames=2,interval=.4,next_delay=.3,
                    quantity_condition={**quantity,'mode':'increment'})
        buy = rule(item+'_buy',label+'数量到上限后购买','%s-min'%item,title,[250,384,475,480],
                   'guild_'+item+'_quantity','guild_'+item+'_confirm',[165,840,548,908],
                   required_source_areas=parts,required_tolerance=30,
                   quantity_condition={**quantity,'mode':'buy'})
        confirm = rule(item+'_confirm','核对金额并确认购买'+label,item+'-confirm',
                       [163,773,256,816],[137,745,288,841],
                       'guild_'+item+'_confirm',following,[111,752,331,822],
                       required_source_areas=[[title[0],410,title[2],438]],required_tolerance=20,
                       color_condition=dict(source_area=[130,754,298,820],hsv=[[35,80,100],[90,255,255]],minimum_fraction=.4),
                       confirmation_condition=dict(label=label+'确认金额',price_area=[278,568,337,610],
                                                   quantity_rule=buy['id'],confidence=.93))
        rules.extend((select,skip,plus,buy,confirm))
    rules.extend((
        rule('close','关闭公会商店','guild',header,[150,312,355,404],
             'guild_close','guild_return',[633,244,700,304],
             required_source_areas=page_parts+[[637,251,688,294]]),
        rule('return','确认回到主界面','main',[649,371,698,415],[610,340,720,450],
             'guild_return','main',action='observe',color_condition=bright,absent_rule=PREFIX+'close'),
    ))
    return rules


if __name__ == '__main__':
    path = ROOT/'config.json'
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    previous = {r['id']:r for r in config['rules'] if r['id'].startswith(PREFIX)}
    rules = presets()
    for entry in rules:
        if entry['id'] in previous:
            entry['enabled'] = previous[entry['id']]['enabled']
            if entry['id'] == PREFIX+'open':
                entry['interval'] = previous[entry['id']]['interval']
    config['rules'] = [r for r in config['rules'] if not r['id'].startswith(PREFIX)] + rules
    path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Guild rules:',len(rules),'Total rules:',len(config['rules']))
