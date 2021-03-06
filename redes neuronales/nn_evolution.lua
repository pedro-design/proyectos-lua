local nn =require('NN_genetico')

local xor_inputs ={{0.1,0.1},{1,0.1},{0.1,1},{1,1}}
local xor_outputs ={{0,1},{1,0},{1,0},{0,1}}
local test={{0.1,2},{0.9,1},{0.87,0.5},{0.3,0.7}}
local arch={2,4,2}-- esta es la arquitectura de la red
local red,act= nn.crear_nn(arch)
local poblacion={}
local poblacionbias = {}
local errores={}
local ind=200
local mejor={}
local mejorl=math.huge
local porcentage=30
nn.hact("leaky_relu") -- las activaciones ocultas
nn.out_act("sigmoid")-- la activacion de la capa final de la red
print(nn.mse(test,xor_outputs),"###")-- test de mse
local elite= math.floor((ind*porcentage)/100) -- porcentajes de selecion

for i=0,ind do
    local p,_,b= nn.crear_nn(arch)
    table.insert(poblacion,p)
	--table.insert(poblacionbias,b)
end
print(#poblacion,elite)
local tmp
local tmppob
local tmpbias
local prediciones={}
local printinterval = 10
local wathcdog = 0
local criterio	 = -0.01
local inicio = os.clock()
while  mejorl ~=0 and  mejorl > criterio   do -- Evoluciona
  errores={}
  tmppob={}
  tmpbias = {}
  if wathcdog == printinterval then
	wathcdog = 0
	io.write("#")
  end
  wathcdog = wathcdog+1
  for i=1,ind do
	 prediciones={}
    for k,v in pairs(xor_inputs) do
      table.insert(prediciones,nn.predecir(xor_inputs[k],poblacion[i],act))


    end

    tmp=nn.mse(xor_outputs,prediciones)
    if tmp< mejorl then
       mejorl=tmp
       mejor=nn.deepCopy(poblacion[i])
	   print("se encontro mejor",tmp)
		prediciones1 = {}

    end
    table.insert(errores,tmp)

  end


  local tablaord,indices =nn.ordenar(errores)

  for ia=0,elite do
	table.insert(tmppob,poblacion[indices[#poblacion-ia]])
  end
  for o = 0,2 do
	local redtt,_= nn.crear_nn(arch)
	table.insert(tmppob,redtt)
  end
  while #tmppob ~=#poblacion do
     local m=#tmppob
     local x1,y1,z1= nn.RNG(1,m),nn.RNG(1,m),nn.RNG(1,m)
	 local c1,c2
	 local b1,b2

	if nn.RNG(0,1) == 1 and mejor ~= nil then
		c1,c2 = nn.cross_over(tmppob[x1],mejor,0.45)

	else
		c1,c2 = nn.cross_over(tmppob[x1],tmppob[y1],0.45)

	end
	 local x1,y1= nn.RNG(1,m),nn.RNG(1,m),nn.RNG(1,m)
     c1=nn.mutar(c1,tmppob[x1],tmppob[y1],0.5,4)
	  local x1,y1= nn.RNG(1,m),nn.RNG(1,m),nn.RNG(1,m)
       c2=nn.mutar(c2,tmppob[x1],tmppob[y1],0.5,4)
     if nn.RNG(0,1)==1 then
        table.insert(tmppob,c1)
     else
        table.insert(tmppob,c2)
     end
  end

  poblacion=nn.deepCopy(tmppob)

end
print(nn.predecir(xor_inputs[1],mejor,act)[1])

local prediciones1={}
for k,v in pairs(xor_inputs) do
   local kkk=nn.deepCopy( nn.predecir(v,mejor,act) )
   table.insert(prediciones1,kkk)
end
tmp=nn.mse(xor_outputs,prediciones1)
print("tardo : ",(os.clock()-inicio)/60)
print('mejor',tmp,mejorl)
for k,v in pairs(prediciones1) do
    print(v[1],v[2])

end
print("----------------------------------")-- imprime los pesos de la mejor red
for k,v in pairs(mejor ) do
	for k1,v1 in pairs(v) do
		for k2,v2 in pairs(v1) do
			io.write(v2)
			io.write(" ")
		end
		io.write("\n")
	end

end
