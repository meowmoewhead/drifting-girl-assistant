"""感知层:数字识别、奖励列表、购买校验、输入证据。

对外只通过各自模块导出,不经由本包名导入,以免拖慢启动。

模块与公开名字(根模块同名的代理壳转发到这些实现):

- ``drift.perception.digit_ocr``      ``DigitReader``、``NumberReading``、``CounterReading``、
  ``parse_counter``、``prepare_counter_crop``、``read_daily_counter``、``read_numeric_views``
- ``drift.perception.reward_list``    ``GreenRewardListButton``
- ``drift.perception.purchase_checks`` ``quantity_outline``、``QuantityTemplates``、``quantity_crop``、
  ``guild_price_crop``、``stock_counter_crop``、``read_stock_counter``、``confirmation_price_crop``、
  ``currency_floor``、``stock_check``、``quantity_check``、``confirmation_check``
- ``drift.perception.boss_evidence``  ``BossInputEvidence``(由其他成员负责)

注意:本文件刻意不 import 上述子模块 —— 只有真正用到时才加载 onnxruntime 与模型。
"""


