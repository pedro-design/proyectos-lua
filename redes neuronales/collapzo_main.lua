local wcoll = require("collapzo")

local configs = {}
-------configuraciones para collapzar la funcion -------------
configs["agua"] = {}
configs["agua"]["color"] ={0, 0.333333, 1}
configs["agua"]["vecinos"] = {"agua","arena","pasto","oasis","bosque"}
--------------------------------------------------------------
configs["arena"] ={}
configs["arena"]["color"] = {1, 1, 0.498039}
configs["arena"]["vecinos"] = {"arena","pasto"}
--------------------------------------------------------------
configs["bosque"] ={}
configs["bosque"]["color"] = {0.666667, 1, 0}
configs["bosque"]["vecinos"] = {"pasto","bosque"}
--------------------------------------------------------------
configs["oasis"] ={}
configs["oasis"]["color"] ={1, 0.666667, 0}
configs["oasis"]["vecinos"] = {"desierto","agua","oasis"}
--------------------------------------------------------------
configs["desierto"] ={}
configs["desierto"]["color"] = {0.282353, 0.854902, 0.588235}
configs["desierto"]["vecinos"] = {"desierto","arena","oasis"}
--------------------------------------------------------------
configs["pasto"] = {}
configs["pasto"]["color"] ={0.333333, 0.666667, 0}
configs["pasto"]["vecinos"] = {"pasto","agua","bosque"}
--////////////////////////////////////////////////////////////////
wcoll.add_rules(configs)
wcoll.entropia = 30
wcoll.set_size(60,60)
-- run the algorithm
local result = wcoll.run()
print(result)
