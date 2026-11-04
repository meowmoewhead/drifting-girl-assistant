"""The real event-shop interruption shares only the guarded first-entry recovery."""
import test_resource_guild_recovery as guild


class ResourceEventPotionRecoveryTests(guild.ResourceGuildRecoveryTests):
    feature = 'event_potion'
    prefix = 'driftinggirl_event_potion_'
    main_asset = 'assets/event-potion/main.png'
    shop_asset = 'assets/event-potion/shop.png'
    battle_fixture = 'tests/fixtures/resource-guild/event-battle.png'
    next_suffix = 'select'
    entry_interval = 60

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # These inherited cases test first-entry recovery, including its hard
        # deadline. Keep cleanup separate without weakening any safety asserts.
        for rule in cls.config['rules']:
            if rule['id'].startswith(cls.prefix):
                rule['skip_on_timeout'] = False
