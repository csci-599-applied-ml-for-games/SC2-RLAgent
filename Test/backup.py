import sc2
from sc2 import run_game, maps, Race, Difficulty, position, Result
from sc2.player import Bot, Computer, Human
from sc2.constants import *
import random
import cv2
import numpy as np
import time
import multiprocessing
import os

HEADLESS = False


class ProtossBot(sc2.BotAI):
    def __init__(self): #设定初始参数
        self.ITERATIONS_PER_MINUTE=260
        self.MAX_WORKERS=38
        self.do_something_after=0
        self.train_data=[]
        self.warpgate_started=False
        self.blink_started=False
        self.proxy_built=False
        self.warpgate_done=False

    def on_end(self, game_result):  #想记录游戏数据，但没有用
        if game_result == Result.Victory:
            np.save(os.path.dirname(os.path.realpath(__file__)) + "/train_data/{}.npy".format(str(int(time.time()))),
                    np.array(self.train_data))

    async def on_step(self, iteration):  #注册所有的动作
        self.iteration = iteration
        await self.distribute_workers()
        await self.build_workers()
        await self.build_pylons()
        await self.build_assimilators()
        await self.offensive_force_buildings()
        await self.build_offensive_force()
        await self.intel()
        await self.attack()
        await self.warp_units()
        await self.build_proxy_pylon()
        await self.chronoboost()
        await self.blink()

    def random_location_variance(self, enemy_start_location):  #随机摆放建筑物，感觉可以再优化
        x=enemy_start_location[0]
        y=enemy_start_location[1]

        x+=((random.randrange(-20,20))/100)*enemy_start_location[0]
        y+=((random.randrange(-20,20))/100)*enemy_start_location[1]

        if x<0:
            x=0
        if y<0:
            y=0
        if x>self.game_info.map_size[0]:
            x=self.game_info.map_size[0]
        if y>self.game_info.map_size[1]:
            y=self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x, y))) #对坐标进行转化
        return go_to

    async def intel(self):  #画图，装逼  其实并没有什么用
        # setup up the data type
        game_data = np.zeros((self.game_info.map_size[1], self.game_info.map_size[0], 3), np.uint8)

        # Prefixed colours for our buildings and units
        unit_colours = {
            NEXUS: [15, (0, 255, 0)],
            PYLON: [3, (20, 235, 0)],
            PROBE: [1, (55, 200, 0)],
            STALKER: [2, (67, 200, 43)],
            ASSIMILATOR: [2, (55, 200, 0)],
            GATEWAY: [3, (200, 100, 0)],
            CYBERNETICSCORE: [3, (150, 150, 0)],
            TWILIGHTCOUNCIL: [5, (255, 0, 0)],
        }

        # Loop over our buildings and draw them
        for unit_type in unit_colours:
            for unit in self.units(unit_type).ready:
                pos = unit.position
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), unit_colours[unit_type][0],
                           unit_colours[unit_type][1], -1)

        main_bases = ["nexus", "commandcenter", "hatchery"]

        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() not in main_bases:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 5, (200, 50, 212), -1)
        for enemy_building in self.known_enemy_structures:
            pos = enemy_building.position
            if enemy_building.name.lower() in main_bases:
                cv2.circle(game_data, (int(pos[0]), int(pos[1])), 15, (0, 0, 255), -1)

        for enemy_unit in self.known_enemy_units:

            if not enemy_unit.is_structure:
                worker_names = ["probe","scv","drone"]

                pos = enemy_unit.position
                if enemy_unit.name.lower() in worker_names:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 1, (55, 0, 155), -1)
                else:
                    cv2.circle(game_data, (int(pos[0]), int(pos[1])), 3, (50, 0, 215), -1)

        # Establishing bars, that represent the game state, into our imagine
        line_max = 50
        mineral_ratio = self.minerals / 1500
        if mineral_ratio > 1.0:
            mineral_ratio = 1.0

        vespene_ratio = self.vespene / 1500
        if vespene_ratio > 1.0:
            vespene_ratio = 1.0

        population_ratio = self.supply_left / self.supply_cap
        if population_ratio > 1.0:
            population_ratio = 1.0

        plausible_supply = self.supply_cap / 200.0

        military_weight = len(self.units(STALKER)) / (self.supply_cap - self.supply_left)
        if military_weight > 1.0:
            military_weight = 1.0

        cv2.line(game_data, (0, 19), (int(line_max * military_weight), 19), (250, 250, 200), 3)  # worker/supply ratio
        cv2.line(game_data, (0, 15), (int(line_max * plausible_supply), 15), (220, 200, 200),3)  # plausible supply (supply/200.0)
        cv2.line(game_data, (0, 11), (int(line_max * population_ratio), 11), (150, 150, 150),3)  # population ratio (supply_left/supply)
        cv2.line(game_data, (0, 7), (int(line_max * vespene_ratio), 7), (210, 200, 0), 3)  # gas / 1500
        cv2.line(game_data, (0, 3), (int(line_max * mineral_ratio), 3), (0, 255, 25), 3)  # minerals minerals/1500

        # flip horizontally to fix visual representation
        self.flipped = cv2.flip(game_data, 0)

        # show the imagine
        if not HEADLESS:
            resized = cv2.resize(self.flipped, dsize=None, fx=2, fy=2)
            cv2.imshow('Intel', resized)
            cv2.waitKey(1)

    async def build_workers(self):  #选择空闲基地建造农民
        if len(self.units(NEXUS))*20>len(self.units(PROBE)):  # 每矿农民补满就不补了
            if len(self.units(PROBE))<self.MAX_WORKERS:
                for nexus in self.units(NEXUS).ready.noqueue:
                    if self.can_afford(PROBE):
                        await self.do(nexus.train(PROBE))

    async def build_pylons(self): #建造水晶塔，且避免建筑影响采矿
        if self.supply_used<=40:
            if self.supply_left<5 and not self.already_pending(PYLON):
                nexus=self.units(NEXUS).ready[-1]
                await self.build(PYLON,near=nexus.position.towards(self.game_info.map_center,10))
        else:
            if self.supply_left<10 and not self.already_pending(PYLON):
                nexus=self.units(NEXUS).ready[-1]
                await self.build(PYLON,near=nexus.position.towards(self.game_info.map_center,10))

    async def build_assimilators(self):  #建造气矿
        for nexus in self.units(NEXUS).ready:
            vespenes=self.state.vespene_geyser.closer_than(15.0,nexus)
            for vespene in vespenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker=self.select_build_worker(vespene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0,vespene).exists:
                    await self.do(worker.build(ASSIMILATOR,vespene))
            
    async def offensive_force_buildings(self): #单兵营开二矿 考虑在建造顺序上更加严谨一点
        if self.units(PYLON).ready.exists:
            pylon = self.units(PYLON).ready.random

            if self.units(GATEWAY).ready.exists and not self.units(CYBERNETICSCORE):
                if self.can_afford(CYBERNETICSCORE) and not self.already_pending(CYBERNETICSCORE):
                    await self.build(CYBERNETICSCORE,near=pylon)
            elif self.units(GATEWAY).amount==1 and self.units(NEXUS).amount<2:
                if self.can_afford(NEXUS):
                    await self.expand_now()
            elif self.units(GATEWAY).amount+self.units(WARPGATE).amount<8: 
                 if self.can_afford(GATEWAY):
                    await self.build(GATEWAY,near=pylon)
                    
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(TWILIGHTCOUNCIL))<1:
                    if self.can_afford(TWILIGHTCOUNCIL) and not self.already_pending(TWILIGHTCOUNCIL):
                        await self.build(TWILIGHTCOUNCIL,near=pylon)
            
            if self.units(CYBERNETICSCORE).ready.exists:
                if len(self.units(ROBOTICSFACILITY))<1:
                    if self.can_afford(ROBOTICSFACILITY) and not self.already_pending(ROBOTICSFACILITY):
                        p=self.game_info.map_center.towards(self.enemy_start_locations[0], 20)
                        await self.build(ROBOTICSFACILITY,near=p)
            
            #升级
            if self.units(CYBERNETICSCORE).ready.exists and self.can_afford(
                    RESEARCH_WARPGATE) and not self.warpgate_started:
                ccore=self.units(CYBERNETICSCORE).ready.first
                await self.do(ccore(RESEARCH_WARPGATE))
                self.warpgate_started = True

            if self.units(TWILIGHTCOUNCIL).ready.exists:
                if self.can_afford(RESEARCH_BLINK) and not self.blink_started:
                    twc = self.units(TWILIGHTCOUNCIL).ready.first
                    await self.do(twc(RESEARCH_BLINK))
                    blink_started = True

    async def build_offensive_force(self): #建造不朽爸爸
         for vr in self.units(ROBOTICSFACILITY).ready.noqueue:
             if self.can_afford(IMMORTAL) and self.supply_left > 0:
                 await self.do(vr.train(IMMORTAL))

    def find_target(self, state):
        if len(self.known_enemy_units) > 0:
            return random.choice(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return random.choice(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]

    #四种攻击选择
        #1.什么都不做
        #2.攻击敌人单位
        #3.攻击敌人建筑物
        #4.攻击敌人出生点 问题：高地视野导致这个变成了送兵，需要手动控制
    async def attack(self):
        if len(self.units(STALKER).idle) > 8:
            choice = random.randrange(0, 4)
            target = False
            if self.iteration > self.do_something_after:
                if choice == 0: #不攻击
                    wait = random.randrange(20, 165)
                    self.do_something_after = self.iteration + wait

                elif choice == 1: #攻击最近的基地
                    if len(self.known_enemy_units) > 0:
                        target = self.known_enemy_units.closest_to(random.choice(self.units(NEXUS)))
                elif choice == 2: #攻击敌人的建筑物
                    if len(self.known_enemy_structures) > 0:
                        target = random.choice(self.known_enemy_structures)
                elif choice == 3: #攻击出生点
                    target = self.enemy_start_locations[0]

                if target: #实现具体攻击
                    for st in self.units(STALKER).idle:
                        await self.do(st.attack(target))
                    for im in self.units(IMMORTAL).idle:
                        await self.do(im.attack(target))
                    for se in self.units(SENTRY).idle:
                        await self.do(se.attack(target))

                y = np.zeros(4) #保存选择 但我没看到有什么效果
                y[choice] = 1
                self.train_data.append([y, self.flipped])


    async def warp_new_units(self, proxy): #折跃追猎者  在考虑同时折跃哨兵
        for warpgate in self.units(WARPGATE).ready:
            abilities = await self.get_available_abilities(warpgate)
            if AbilityId.WARPGATETRAIN_STALKER in abilities:
                pos = proxy.position.to2.random_on_distance(4)
                placement = await self.find_placement(AbilityId.WARPGATETRAIN_STALKER, pos, placement_step=1)
                if placement is None:
                    return
                if self.units(ROBOTICSFACILITY).amount<1 or (not self.units(ROBOTICSFACILITY).noqueue):
                    await self.do(warpgate.warp_in(STALKER, placement))
                    await self.do(warpgate.warp_in(SENTRY,placement))
                
    async def warp_units(self):
        if self.proxy_built and self.units(WARPGATE).ready:
            proxy = self.units(PYLON).closest_to(self.enemy_start_locations[0])
            await self.warp_new_units(proxy)

    async def build_proxy_pylon(self): #野水晶，加快接兵速度
        if self.units(CYBERNETICSCORE).amount >= 1 and not self.proxy_built and self.can_afford(PYLON):
            p = self.game_info.map_center.towards(self.enemy_start_locations[0], 20)
            await self.build(PYLON, near=p)
            self.proxy_built = True

    async def chronoboost(self): #使用星空加速来加快研究速度
        if not self.units(CYBERNETICSCORE).ready.exists:
            bn=self.units(NEXUS).ready.first
            if not bn.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                for nexus in self.units(NEXUS).ready:
                    abilities = await self.get_available_abilities(nexus)
                    if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        for nexus in self.units(NEXUS).ready:
                            await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, bn))
        elif self.units(CYBERNETICSCORE).ready.exists and not self.units(WARPGATE).exists:
            ccore = self.units(CYBERNETICSCORE).ready.first
            if not ccore.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                for nexus in self.units(NEXUS).ready:
                    abilities = await self.get_available_abilities(nexus)
                    if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        for nexus in self.units(NEXUS).ready:
                            await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, ccore))
        elif self.units(TWILIGHTCOUNCIL).ready.exists:
            twlc = self.units(TWILIGHTCOUNCIL).ready.first
            if not twlc.has_buff(BuffId.CHRONOBOOSTENERGYCOST):
                for nexus in self.units(NEXUS).ready:
                    abilities = await self.get_available_abilities(nexus)
                    if AbilityId.EFFECT_CHRONOBOOSTENERGYCOST in abilities:
                        for nexus in self.units(NEXUS).ready:
                            await self.do(nexus(AbilityId.EFFECT_CHRONOBOOSTENERGYCOST, twlc))

    async def has_ability(self, ability, unit): #判断技能是否可用
        abilities = await self.get_available_abilities(unit)
        if ability in abilities:
            return True
        else:
            return False


    async def blink(self): #当护盾小于20的时候往后闪烁，规避火力
        for stalker in self.units(STALKER):
            has_blink = await self.has_ability(EFFECT_BLINK_STALKER, stalker)
            threatsClose = self.known_enemy_units.filter(lambda x: x.can_attack_ground).closer_than(5, stalker)
            if threatsClose and stalker.shield < 20:
                escape_location = stalker.position.towards(self.game_info.map_center, 4)
                if has_blink:
                    await self.do(stalker(EFFECT_BLINK_STALKER, escape_location))


def game():
    run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, ProtossBot()),
        Computer(Race.Protoss, Difficulty.Hard)
    ], realtime=False)

game()