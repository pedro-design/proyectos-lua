--nn.refresh()


local nn =require('NN_genetico')
local x,y = {{0,1},{1,1},{0,0},{1,0}},{{1},{0},{0},{1}}
--local test_data =  nn.crear_ventana(x,5)
local arch={2,1,0.3} -- arquitectura de la red
local numero_de_individuos = 100
local modo_rnn = false -- si decimos que no, solo predecimos los datos de manera clasica, si es verdadero, nos movemos a traves de una serie de longitud N y sumamos todas las predicciones
local porcentaje_elite = 30 -- porcentage de individuos elite
local hyperparametro = 0.3
local tb = 100
local modo = "NEAT"

nn.hact("leaky_relu")-- activacion de las capas ocultas
nn.out_act("sigmoid")-- activacion de la salida

-- init fitness table
nn.iniciar_individuos(arch,numero_de_individuos,porcentaje_elite,hyperparametro,tb,modo)
nn.predecir_individuo(x[1],false)
local step_p = 0
local generacion =0
while generacion<900 do
    if step_p == 100 then
		print(generacion)
		step_p=0
	end
	step_p=step_p+1
    for individuo = 1,numero_de_individuos do
        local loss = 0
        local preds = {}
        for k,v in pairs(x) do
           -- print(individuo)
            table.insert(preds,nn.predecir_individuo(v,false) )
        end
        local works = true
        for k,v in pairs(preds) do
            if v == nil then
                works = false
            end
        end
        if #y ~= #preds then
            works = false
        end
        if works == true  and nn.mse(y,preds) ~= nil then
            loss =nn.mse(y,preds)*10
			if loss ==0  then
			--	generacion = 20000
			end
        else
            loss = 100
        end
        --invertimos el valor de loss, ya que en modo agente ml, mientras mayor sea el fitness, en teoria menor seria la loss
        nn.establecer_recompenza(100 -loss) -- 100 es nuestro tope de fitness
        nn.siguiente_individuo(false)

    end
   -- print("gen")
   --print("G")
   nn.crear_nueva_generacion(0.3,0.2,0.4,false)

   generacion = generacion+1
end
print("best")
local preds = {}
for k,v in pairs(x) do
    table.insert(preds,nn.predecir_mejor_individuo(v,false) )
end
nn.tprint(preds)

print("Fin")
nn.print_mejor()
