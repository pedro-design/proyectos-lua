local nn =require('NN_genetico')
nn.refresh()
-- tests the functions above
local file = 'palabras.txt'
local lines = nn.lines_from(file)
local x,y = {},{}
-- print all line numbers and their contents
for k,v in pairs(lines) do
	local pp =  v:split(".")
	--for k1,v1 in pairs(pp) do io.write(" ",v1) end
	--io.write("\n")
	table.insert(x,nn.to_bytes_table(pp[1]))
	table.insert(y,{pp[2]})

end
--for k,v in pairs(x) do
	--for k1,v1 in pairs(v) do io.write(" ",v1) end
	--io.write("\n")
--end



local x,y = {{0,1},{1,1},{0,0},{1,0}},{{1},{0},{0},{1}}
local test_data =  nn.crear_ventana(x,5)
local arch={2,8,1} -- arquitectura de la red
local numero_de_individuos = 50
local criterio = 0.25 -- este es el criterio de el error minimo para detenerse, si es negativo, no se detiene hasta que ya no se mejore mas el error
local modo_rnn = false -- si decimos que no, solo predecimos los datos de manera clasica, si es verdadero, nos movemos a traves de una serie de longitud N y sumamos todas las predicciones
local porcentaje_elite = 30 -- porcentage de individuos elite
local  minimo_hyperparametro = 0.01
local hyperparametro = 1

nn.hact("leaky_relu")-- activacion de las capas ocultas
nn.out_act("leaky_relu")-- activacion de la salida

local lua_se_bloquea_sin_wait = false

mejor,act,b = nn.optimizador_genetico_con_datasets(x,y,arch,numero_de_individuos,criterio,modo_rnn,30,lua_se_bloquea_sin_wait,minimo_hyperparametro,hyperparametro)
local out = nn.predecir(x[1],mejor,act,b,modo_rnn)
