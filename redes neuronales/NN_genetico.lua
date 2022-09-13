
local NN = {}
local h_act = 0
local out_actf = 0
local huge = math.huge
local rng = math.random
local exp = math.exp
local sqrt= math.sqrt
local tanh = math.tanh
local modo_arch = "normal"
local fitness = {}
local poblacion_agentes = {}
local poblacion_agentes_biases = {}
local init = false
local actual = 1
local act_scheme = {}
local global_arch = {}
local elite_indv = 1
local mejor_agente = {}
local mejor_agente_biases ={}
local tba = 2
local mejor_fitness_global = -math.huge
local batch_actual = 1
local batches_disponibles = 1
local prev_wd_fitness = -math.huge
local config = {"wd","mr","Wms","addrt","crossrt"}
config["wd"]= 20
config["mr"] = 0.4
config["Wms"]= 3
config["addrt"] = 0.4
config["crossrt"] = 0.5 -- Not used-- 1/2

function NN.change_config(x)
	config = x
end

function NN.get_config()
	return config
end

function NN.r3(x,y)
	return (x*y)/50

end

function NN.memorizar(f)
	local mem ={}
	setmetatable(mem,{mode="kv"})
	return function (x)
		local r = mem[x]
		if r == nil then
			r = f(x)
		end
		return r
	end
end


math.randomseed(os.time())

function NN.crear_ventana(array,t)
	local indice= 1

	local out = {}
	local tmp = {}

	while indice< #array  and array[indice+t-1]~=nil  do
		tmp = {}
		for i=indice,indice+t-1 do
			table.insert(tmp,array[i])
			--	print(i)
		end
		while #tmp ~= t do

			table.insert(tmp,0)
		end
		--	print("SDS")
		table.insert(out,tmp)
		indice = indice+1
	end

	return out

end


function NN.file_exists(file)
	local f = io.open(file, "rb")
	if f then f:close() end
	return f ~= nil
end

function NN.lines_from(file)
	if not NN.file_exists(file) then print("file dosent exist") return {} end
	local lines = {}
	for line in io.lines(file) do
		lines[#lines + 1] = line
	end
	return lines
end


function NN.media(x)
	local suma = 0
	for _,v in pairs(x) do
		suma = suma+v
	end
	return suma/ #x
end


function NN.suma(x)
	local suma = 0
	for _,v in pairs(x) do
		suma = suma+v
	end
	return suma
end


function NN.refresh()
	math.randomseed(os.time())
	math.random(0,10)
end

function NN.Norm(x)
	local max,min=0,math.huge
	local tmp = {}
	for k,v in pairs(x) do
		if v <min then
			min = v
		end
		if v> max then
			max = v
		end
		table.insert(tmp,0)
	end
	for k,v in pairs(x) do
		tmp[k] = ((v - min)/ (max - min))--0.5)/0.5
	end
	return tmp
end
------------------------funciones de activacion----------------


------------------prueva de compatibilidad---------------------
local test_names = {"math.huge","math.random","io.write"}
local test_set = {math.huge,math.random,io}
for k,v in pairs(test_set) do
	if v == nil then
		print(test_names[k]," is not compatible ")
	end
end






--------------------------------------------------------

function NN.sigmoid(x)
	return 1 / (1 + math.exp(-x))
end

function NN.soft_max(x)
	if type(x)~= "table" then
		print("la entrada debe ser una tabla")
	end

	local suma = 0
	for _,v in pairs(x) do
		suma = suma+math.exp(v)
	end


	local probs = {}
	for k,v in pairs(x) do
		v = math.exp(v)
		table.insert(probs,v/suma)

	end
	return probs
end


function NN.leaky_relu(x,scale)
	if x >0 then
		return x
	else
		return x*scale
	end
end

function NN.relu(x)
	if x <0 then
		return 0
	else
		return x
	end
end
-----------------------------Funciones para evolucionar------------------
function NN.RNG(x,y)

	return rng(x,y)
end
function NN.out_act(x)
	out_actf = x
end
function NN.hact(x)
	h_act = x
end


function NN.cross_over_bias(x,y,m)
	local x1 = NN.deepCopy(x)
	local y1 = NN.deepCopy(y)
	for k,v in pairs(x) do
		for k1,v1 in pairs(v) do
			local n = NN.noise(10)
			if NN.RNG(0,100)/100 < m then
				y1[k][k1]=x[k][k1]
				x1[k][k1]=y[k][k1]
			else
				x1[k][k1]=x[k][k1]
				y1[k][k1]=y[k][k1]
			end
		end
	end
	return x1,y1
end



function NN.deepCopy(original)-- con esto clonamos las tablas
	local copy = {}
	for k, v in pairs(original) do
		if type(v) == "table" then
			v = NN.deepCopy(v)
		end
		copy[k] = v
	end
	return copy
end

function NN.mse(x,y)
	local tmp =0
	for k=1,#y  do
		for k1=1,#y[k]  do
			if x[k][k1]~= nil and y[k][k1]~= nil then
				tmp=tmp + ((x[k][k1]-y[k][k1])^2)

			end
		end
	end
	return tmp/#x

end
----------dot product para las neuronas
function NN.dot(x,y)
	local sum =0
	if type(x)== 'table' and type(y) == 'table' then
		if #x == #y then
			for i=1,#x,1 do
				sum = sum + (x[i]*y[i])
			end
		else
			return nil
		end
		--	print("not a table")
	elseif type(x)== 'table' and type(y) ~= 'table'  then
		for i=1,#x,1 do
			sum=sum+ (x[i]*y)

		end

	end
	return sum
end
--------------------------
function NN.media_2(x,y)
	local tmp1
	local tmp2
	local grad = 0
	local grad2 = 0
	local x1 = NN.deepCopy(x)
	local y1 = NN.deepCopy(y)
	if type(x)== 'table' and type(y) == 'table' and #x == #y then
		for k,v in pairs(x) do
			if type(v)== 'table'and type(y[k]) == 'table' then
				for k1,v1 in pairs(v) do
					for k2,v2 in pairs(v1) do

						-------------------------
						tmp1 = v2
						tmp2 = y[k][k1][k2]
						x1[k][k1][k2] = (tmp1+tmp2)/2

						--print("muto ",tmp1 ," a ",tmp2)

					end
				end
			end
		end
	else
		print("se requiere que sean nn las 2 entradas")

	end
	return x1
end



function NN.ordenar(x)
	local ordenados={}
	local indices={}
	local maximo=-huge
	local tmpindice
	local fin=false
	for k,v in pairs(x) do
		if v>maximo then
			maximo=v
			tmpindice=k
		end
	end
	table.insert(ordenados,maximo)
	table.insert(indices,tmpindice)
	while #ordenados ~= #x and not fin do
		maximo=-huge
		for k,v in pairs(x) do
			if v>=maximo and v<ordenados[#ordenados] then
				maximo=v
				tmpindice=k
			end
		end
		if maximo ~= -huge then
			--  print(maximo,tmpindice)
			table.insert(ordenados,maximo)
			table.insert(indices,tmpindice)
		else fin= true
		end
	end
	return ordenados,indices
end

function NN.cross_over(x,y,mr)
	local tmp1
	local tmp2
	local grad = 0
	local grad2 = 0
	local x1 = NN.deepCopy(x)
	local y1 = NN.deepCopy(y)

	if type(x)== 'table' and type(y) == 'table' and #x == #y then
		for k,v in pairs(x) do
			if type(v)== 'table'and type(y[k]) == 'table' then
				for k1,v1 in pairs(v) do
					for k2,v2 in pairs(v1) do

						-------------------------
						if rng(1,1000)/1000< mr then
							tmp1 = v2
							tmp2 = y[k][k1][k2]
							grad = (tmp2-tmp1)*0.001

							grad2 = (tmp1-tmp2)*0.001

							x1[k][k1][k2] = tmp2-grad--+ ((rng(1,1000)/1000 )-rng(1,1000)/1000)*0.0001--grad2
							y1[k][k1][k2] = tmp1-grad2--+ ((rng(1,1000)/1000 )-rng(1,1000)/1000)*0.0001--grad
							--print("muto ",tmp1 ," a ",tmp2)
						end
					end
					--local normlx = NN.Norm(v1)
					--local normly = NN.Norm(y1[k][k1])
					--	for k2,v2 in pairs(v1) do
					--	print(normlx[k2],normly[k2])
					--	print(x1[k][k1][k2]*normlx[k2] ,y1[k][k1][k2]*normly[k2])
					--	wait()
					--	x1[k][k1][k2] = x1[k][k1][k2] * normlx[k2]
					--	y1[k][k1][k2]=y1[k][k1][k2] * normly[k2]
					--end
				end
			end
		end
	else
		print("se requiere que sean nn las 2 entradas")

	end
	return x1,y1
end
-----------------------------Handle de las redes neuronales ------------------
function NN.crear_nn_normal(capas)
	local pesos ={}
	local activaciones ={}
	local biases = {}
	local indice_capa = 1
	local tmp={}
	for i=1,#capas,1 do
		table.insert(activaciones, {})
		table.insert(pesos,{})
		table.insert(biases,{})
	end
	for k,v in pairs(capas) do
		if k==1 then
			for i=1,v,1 do
				local tmp2={}
				for o=1,capas[k] do
					--local w =(NN.noise(10)+NN.noise(10)+NN.noise(10))--*math.sqrt(2 /capas[k-1]+capas[k])
					local w = (((rng(1,1000)/1000)+(rng(1,1000)/1000))+((rng(1,1000)/1000)+(rng(1,1000)/1000)) ) -- un peso para la conexion de -4 a 4
					if rng(0,1) == 1 then w = w*-1 end
					table.insert(tmp2,w)
				end
				table.insert(pesos[k],tmp2)
				table.insert(activaciones[k],0)
				table.insert(biases[k],NN.noise(10))
			end
		else
			for i=1,v,1 do
				local tmp2={}
				for o=1,#pesos[k-1] do
					local w = (((rng(1,1000)/1000)+(rng(1,1000)/1000))+((rng(1,1000)/1000)+(rng(1,1000)/1000)) ) -- un peso para la conexion de -4 a 4
					if rng(0,1) == 1 then w = w*-1 end
					table.insert(tmp2,w)
				end
				table.insert(pesos[k],tmp2)
				table.insert(activaciones[k],0)
				table.insert(biases[k],NN.noise(10))
			end
		end
	end
	-- table.insert(pesos,tmp)
	return pesos,activaciones,biases
end


function string_split (str,sep)
	local sep, fields = sep or ":", {}
	local pattern = string.format("([^%s]+)", sep)
	str:gsub(pattern, function(c) fields[#fields+1] = c end)
	return fields
end

function NN.to_bytes_table (str)
	local t = {}
	for i = 1, #str do
		t[i] = string.byte(str:sub(i, i))
	end
	return t
end


function NN.predecir(x,pesos,b,activaciones1,rnn)
	local indice = 1
	local outsize = #pesos[#pesos]
	local step = #pesos[1]-outsize
	local sum = {}
	local flatten = {}
	-- k,v in pairs(x) do print(v) end
	local activaciones = NN.deepCopy(activaciones1) -- problemas de memoria
	if not rnn then
		if #x ~= step+outsize then print("tamaños no concuerdan,",#x,#pesos[1])  return {0} end
		for k=1,#pesos do
			local v = pesos[k]
			if k ==1  then

				for k1=1,#v do
					local v1 = v[k1]
					if h_act == "tanh"  then
						activaciones[k][k1] =tanh(NN.dot(x,v1))
					elseif h_act =="sigmoid"  then
						activaciones[k][k1] =NN.sigmoid(NN.dot(x,v1))
					elseif h_act =="relu"  then
						activaciones[k][k1] =NN.relu(NN.dot(x,v1))
					elseif h_act =="leaky_relu"  then
						activaciones[k][k1] =NN.leaky_relu(NN.dot(x,v1),0.05)

					else
						activaciones[k][k1] =(NN.dot(x,v1))
					end
					activaciones[k][k1]= activaciones[k][k1]+ b[k][k1]
				end
				indice = indice+1
			elseif k~= #activaciones then
				for k1=1,#v do
					local v1 = v[k1]
					if h_act == "tanh"  then
						activaciones[k][k1] = tanh(NN.dot(activaciones[k-1],v1))
					elseif h_act =="sigmoid"   then
						activaciones[k][k1] =NN.sigmoid(NN.dot(activaciones[k-1],v1))
					elseif h_act =="relu"  then
						activaciones[k][k1] =NN.relu(NN.dot(activaciones[k-1],v1))
					elseif h_act =="leaky_relu"  then
						activaciones[k][k1] =NN.leaky_relu(NN.dot(activaciones[k-1],v1),0.05)
					else
						activaciones[k][k1] =(NN.dot(activaciones[k-1],v1)) --linear no sirve ,es muy inestable
					end
					activaciones[k][k1]= activaciones[k][k1]+ b[k][k1]

				end
			else
				for k1=1,#v do
					local v1 = v[k1]
					if out_actf == "tanh"  then
						activaciones[k][k1] = tanh(NN.dot(activaciones[k-1],v1))
					elseif out_actf =="sigmoid"  then
						activaciones[k][k1] =NN.sigmoid(NN.dot(activaciones[k-1],v1))
					elseif out_actf =="relu"  then
						activaciones[k][k1] =NN.relu(NN.dot(activaciones[k-1],v1))
					elseif out_actf =="leaky_relu"  then
						activaciones[k][k1] =NN.leaky_relu(NN.dot(activaciones[k-1],v1),0.05)
					else
						activaciones[k][k1] =(NN.dot(activaciones[k-1],v1))--linear no sirve ,es muy inestable
					end
					activaciones[k][k1]= activaciones[k][k1]+ b[k][k1]

				end
				--	print("fin")
			end
			indice = indice+1
			--	indice = indice+1



		end
		if out_actf =="soft_max" then
			return NN.soft_max( activaciones[#activaciones])
		else
			return activaciones[#activaciones]
		end



	else --conectamos la ultima salida a la entrada actual, modo rnn

		for i=1,#activaciones[#activaciones] do
			table.insert(sum,0)
			table.insert(flatten,0)
		end

		local steps = NN.crear_ventana(x,step)
		--print("modo rnn",#steps,#x)
		activaciones = NN.deepCopy(activaciones1)
		local op = 1

		for k9,v9 in pairs(steps) do
			op = 1
			--pass = v9
			while #v9 ~= step+outsize do
				table.insert(v9,flatten[ op])
				op =  op+1
			end
			--print(v9[1],v9[2],v9[3],"$$$$$$$$$$$",#v9,step)
			if #v9 ~= step+outsize then print("tamaños no concuerdan,rnn ,",#v9,step)  return {0} end
			for k=1,#pesos do
				local v = pesos[k]
				if k ==1  then

					for k1=1,#v do
						local v1 = v[k1]
						if h_act == "tanh"  then
							activaciones[k][k1] =tanh(NN.dot(v9,v1))
						elseif h_act =="sigmoid"  then
							activaciones[k][k1] =NN.sigmoid(NN.dot(v9,v1))
						elseif h_act =="relu"  then
							activaciones[k][k1] =NN.relu(NN.dot(v9,v1))
						elseif h_act =="leaky_relu"  then
							activaciones[k][k1] =NN.leaky_relu(NN.dot(v9,v1),0.05)

						else
							activaciones[k][k1] =(NN.dot(v9,v1))
						end
						activaciones[k][k1]= activaciones[k][k1]+ b[k][k1]

						--	print(activaciones[k][k1],"ACT",k,k1)
					end

				elseif k~= #activaciones then
					for k1=1,#v do
						local v1 = v[k1]
						if h_act == "tanh"  then
							activaciones[k][k1] =tanh(NN.dot(activaciones[k-1],v1))
						elseif h_act =="sigmoid"   then
							activaciones[k][k1] =NN.sigmoid(NN.dot(activaciones[k-1],v1))
						elseif h_act =="relu"  then
							activaciones[k][k1] =NN.relu(NN.dot(activaciones[k-1],v1))
						elseif h_act =="leaky_relu"  then
							activaciones[k][k1] =NN.leaky_relu(NN.dot(activaciones[k-1],v1),0.05)
						else
							activaciones[k][k1] =(NN.dot(activaciones[k-1],v1)) --linear no sirve
						end
						activaciones[k][k1]= activaciones[k][k1]+ b[k][k1]

						--	print(activaciones[k][k1],"ACT",k,k1)
					end

				else
					for k1=1,#v do
						local v1 = v[k1]
						if out_actf == "tanh"  then
							activaciones[k][k1] = math.tanh(NN.dot(activaciones[k-1],v1))
						elseif out_actf =="sigmoid"  then
							activaciones[k][k1] =NN.sigmoid(NN.dot(activaciones[k-1],v1))
						elseif out_actf =="relu"  then
							activaciones[k][k1] =NN.relu(NN.dot(activaciones[k-1],v1))
						elseif out_actf =="leaky_relu"  then
							activaciones[k][k1] =NN.leaky_relu(NN.dot(activaciones[k-1],v1),0.05)
						else
							activaciones[k][k1] =(NN.dot(activaciones[k-1],v1))
						end
						activaciones[k][k1]= activaciones[k][k1]+ b[k][k1]

						--	print(activaciones[k][k1],"ACT",k,k1)
					end
					--print(activaciones[k][k1],"ACT",k,k1)
					--	print("fin")

				end
				--indice = indice+1
				--	indice = indice+1



				--return activaciones[#activaciones]

			end

			for kk,vv in pairs(activaciones[#activaciones]) do
				sum[kk] = sum[kk]+vv
				flatten[kk] = vv
				--	print(vv)
			end
			--	print(activaciones[#activaciones][1],"OUT")

		end

	end
	return sum
end

function NN.noise(x)
	local sum = 0
	for i = 1,x do
		sum = sum + (((rng(0,1000)/1000)-0.5)/0.5 )/i
	end
	return sum
end

function NN.mutar_bias(x,y,z,grad,res)
	local tmp=NN.deepCopy(x)
	local tn = 0
	for k,v in pairs(x) do
		for k1,v1 in pairs(v) do
			--for k2,v2 in pairs(v1) do
			---	tn = 0
			--	if rng(0,1000)/1000 <cr then
			tn= NN.noise(res)

			tmp[k][k1]=v1+ (( y[k][k1] - z[k][k1])*grad)+(tn*grad)

			--	end

			--end
		end
	end
	return tmp
end

function NN.mutar(x,y,z,grad,res)
	local tmp=NN.deepCopy(x)
	local tn = 0
	for k,v in pairs(x) do
		for k1,v1 in pairs(v) do
			for k2,v2 in pairs(v1) do
				---	tn = 0
				--	if rng(0,1000)/1000 <cr then
				tn= NN.noise(res)

				tmp[k][k1][k2]=v2+ (( y[k][k1][k2] - z[k][k1][k2])*grad)+(tn*grad)

				--	end

			end
		end
	end
	return tmp
end

function NN.old_mutar(x,mr)
	local tmp1=0
	local x1 = NN.deepCopy(x)
	if type(x)== 'table' then
		for k,v in pairs(x) do
			if type(v)== 'table' then
				for k1,v1 in pairs(v) do
					for k2,v2 in pairs(v1) do

						if rng(1,1000)/1000> mr then
							print(v2,x1[k][k1][k2])
							tmp1 = v2+ ((rng(1,1000)/1000)-(rng(1,1000)/1000))*0.5

							x1[k][k1][k2] = tmp1

							--print("muto ",tmp1 ," a ",tmp2)

						end
					end
				end
			end
		end
	else
		print("se requiere que sean nn las 2 entradas")
	end
	return x1
end

function NN.crear_desendientes(tmppobb,tmpbias,tamano_de_poblacion,hyperparametro,addrt,mtw)
	-- DEBUG NN.tprint(tmppob)
	local tmppob = NN.deepCopy(tmppobb)
	--	if modo_arch == "normal" then
	--	local tmppob = NN.deepCopy(tmppob)
	--end

	if modo_arch == "normal" then
		local tmpbias = NN.deepCopy(tmpbias)
		while #tmppob ~=tamano_de_poblacion do
			local m=#tmppob
			local x1,y1,z1= NN.RNG(1,m),NN.RNG(1,m),NN.RNG(1,m)
			local c1,c2
			local b1,b2

			if NN.RNG(0,1) == 1 and mejor ~= nil and mejorb~= nil then
				c1,c2 = NN.cross_over(tmppob[x1],mejor,hyperparametro)
				b1,b2= NN.cross_over_bias(tmpbias[x1],mejorb,hyperparametro)
			else
				c1,c2 = NN.cross_over(tmppob[x1],tmppob[y1],hyperparametro)
				b1,b2=NN.cross_over_bias(tmpbias[x1],tmpbias[y1],hyperparametro)
			end
			local x1,y1= NN.RNG(1,m),NN.RNG(1,m)
			c1=NN.mutar(c1,tmppob[x1],tmppob[y1],hyperparametro,4)
			b1= NN.mutar_bias(b1,tmpbias[x1],tmpbias[y1],hyperparametro,4)
			local x1,y1= NN.RNG(1,m),NN.RNG(1,m)
			c2=NN.mutar(c2,tmppob[x1],tmppob[y1],hyperparametro,4)
			b2=NN.mutar_bias(b2,tmpbias[x1],tmpbias[y1],hyperparametro,4)
			if NN.RNG(0,1)==1 then
				table.insert(tmppob,c1)
				table.insert(tmpbias,b1)
			else
				table.insert(tmppob,c2)
				table.insert(tmpbias,b2)
			end
		end

		return tmppob,tmpbias
	elseif modo_arch == "NEAT" then
		-- print("E##############")
		--   NN.tprint(tmppob)
		--print("LOCK")
		while #tmppob ~=tamano_de_poblacion and not (#tmppob>tamano_de_poblacion) do
			--  NEAT_crossover
			local m=#tmppob
			local x1,y1= NN.RNG(1,m),NN.RNG(1,m)
			if x1== y1 then
				while x1 == y1 and m~=1  do --no queremos que sean iguales los indices
					x1,y1= NN.RNG(1,m),NN.RNG(1,m)
				end
			end
			 -- print("nueva gen ",#tmppob[x1]["neuronas"],#tmppob[y1]["neuronas"],#tmppob)

			-- NN.NEAT_crossover(net1,net2,mr,addrt,mw)
			local desendiente   -- valores por defecto
			if NN.RNG(0,1) == 1 and mejor ~= nil  then
				desendiente,added =  NN.NEAT_crossover(tmppob[x1],mejor,hyperparametro,addrt,mtw)
			else
				desendiente,added =  NN.NEAT_crossover(tmppob[x1],tmppob[y1],hyperparametro,addrt,mtw)
			end
			if added == true then
				fitness[#tmppob] = fitness[#tmppob] -100 -- penalizamos agregar neuronas
			end

			table.insert(tmppob,desendiente)

		end
		return tmppob
	end
end
---------------------------implementacion del algorithmo NEAT ------------------ :D # bueno para motores de juegos como Roblox o LÖVE
function NN.tprint (tbl, indent)
	if not indent then indent = 0 end
	for k, v in pairs(tbl) do
		formatting = string.rep(" ", indent) .. k .. ": "
		if type(v) == "table" then
			print(formatting)
			NN.tprint(v, indent+1)
		else
			print(formatting .. tostring(v))
		end
	end
end

local charset = {}  do -- [0-9a-zA-Z]
	for c = 48, 57  do table.insert(charset, string.char(c)) end
	for c = 65, 90  do table.insert(charset, string.char(c)) end
	for c = 97, 122 do table.insert(charset, string.char(c)) end
end

function NN.randomString(length)
	if not length or length <= 0 then return '' end
	-- math.randomseed(os.clock()^5)
	return NN.randomString(length - 1) .. charset[math.random(1, #charset)]
end
--


--cross over en NEAT
function NN.NEAT_crossover(net1,net2,mr,addrt,mw)
	--print("SSSSMNEATCROSSOVER")
	local red_resultante = {}
	local recuento_neuronas_internas = 0
	local capas = 1
	local new_n = false
	if net1["neuronas"]== nil or net2["neuronas"] == nil then
		print("ERROR", net1["neuronas"] , " net2 ",net2["neuronas"])
	end
	red_resultante["neuronas"] = {}
	red_resultante["Nodos_Flatten"] = {}
	local busqueda_nodos_iguales = {} --buscamos nodos que no existan en ambas redes
	for k,v in pairs(net1["neuronas"]) do -- escaneamos primero una red
		if busqueda_nodos_iguales[v["ID"]] == nil then
			table.insert(busqueda_nodos_iguales,v["ID"])
			busqueda_nodos_iguales[v["ID"]] = 0
		else
			busqueda_nodos_iguales[v["ID"]] = 1
		end

	end
	for k,v in pairs(net2["neuronas"]) do -- luego la otra
		if busqueda_nodos_iguales[v["ID"]] == nil then
			table.insert(busqueda_nodos_iguales,v["ID"])
			busqueda_nodos_iguales[v["ID"]] = 0
		else
			busqueda_nodos_iguales[v["ID"]] = 1
		end
	end
	-- red_resultante["Nodos_Flatten"] = {}
	-- ahora insertamos las neuronas en la red resultante

	--el problema es que se agregan 2 veces las salidas y las entradas

	for k,v in pairs(net1["neuronas"]) do --agragamos todas las neuronas de la primera red
		if red_resultante["neuronas"][k] == nil then
			table.insert(red_resultante["neuronas"],v)
		end

	end
	for k,v in pairs(net2["neuronas"]) do  -- agregamos las neuronas que nos falten
		if red_resultante["neuronas"][k]~= nil and red_resultante["neuronas"][k]["ID"]  == nil then
			local found = false
			for kk,vv in pairs(red_resultante["neuronas"]) do
				if v["ID"] == vv["ID"] then
					found = true -- iteramos y buscamos si hay una coincidencia
				end
			end
			if found == false then
				table.insert(red_resultante["neuronas"],v)
			end
		end


	end
	--contamos las neuronas internas
	local hay_hl = false
	local indices_h={}
	for k,v in pairs(red_resultante["neuronas"]) do
		if v["tipo"] ==2 then
			--   print(v["layer"],capas)
			if v["layer"]> capas then
				capas = v["layer"]
			end
			table.insert(indices_h,k)
			if hay_hl==false then hay_hl=true end --check
			recuento_neuronas_internas=recuento_neuronas_internas+1
		end
	end
	--si hay provabilidad , removemos 1 neurona entera


	--print("DEBUG ",recuento_neuronas_internas,capas)
	-- despues de estos tenemos todas las neuronas disponibles
	-- como funciona
	-- red1          12
	-- red2          1234    # es que se efectuo un cross_over  + es que se añadieron nodos
	-- resultado     #12 + 34
	if math.random(0,1000)/1000 < addrt  then
		new_n = true
		--  table.insert(red_resultante["Nodos_Flatten"],neurona["ID"])
		local mask = {}
		local candidatos = {}
		local capa = capas+1
		for k,v in pairs(red_resultante["neuronas"]) do
			if v["layer"]~= capa and ( (v["layer"] <capa) and v["layer"]~= -1)   then -- mantenemos una relacion linear
				table.insert(mask,math.random(0,1))
				table.insert(candidatos,v["ID"])

			elseif v["layer"]== -1 and  v["layer"]~= capa then
				table.insert(mask,math.random(0,1))
				table.insert(candidatos,v["ID"])
			end
		end
		-- iteramos y generamos conexiones

		local final_candidatos = {}
		--print(capas)
		local neurona = NN.agregar_neurona_NEAT(capas+1,nil,recuento_neuronas_internas+1)-- agregamos una neurona mas
		table.insert(red_resultante["neuronas"],neurona)
		for candidato,valor in pairs(mask) do
			if valor == 1 then
				--    print("conexion de ",neurona["ID"] , " a ", candidatos[candidato],neurona["Inovations"] )
				table.insert(neurona["Inovations"],candidatos[candidato])
				local W = ((rng(1,1000)/1000)-(rng(1,1000)/1000)) -- un peso para la conexion de -2 a 2
				table.insert(neurona["W"],W)
				if  neurona["B"] == 0 then
					local B = ((rng(1,1000)/1000)-(rng(1,1000)/1000)) -- un peso para el bias
					neurona["B"] = B
				end
			end
		end

	elseif  math.random(0,1000)/1000 < addrt and #indices_h>1 then
		local nodo_a = math.random(1,#indices_h)
		table.remove(red_resultante["neuronas"][nodo_a])-- lo removemos


	end
	--aplicamos cross over
	for k,neurona in pairs(red_resultante["neuronas"]) do
		if busqueda_nodos_iguales[neurona["ID"]]~= nil and busqueda_nodos_iguales[neurona["ID"]]==1 then
			-- cross over  y su probalididad
			if math.random(0,1000)/1000 < mr then
				if math.random(0,1) == 1 then -- escojemos 1 nodo de las 2 redes
					local nodo = {}
					for kk,vv in pairs(net2["neuronas"]) do
						if vv["ID"] == neurona["ID"] then
							nodo = vv
						end
					end
					if nodo["ID"]~= nil then
						red_resultante["neuronas"][k] = nodo
					end


				else
					-- usamos la neurona del la otra red
					local nodo = {}
					for kk,vv in pairs(net1["neuronas"]) do
						if vv["ID"] == neurona["ID"] then
							nodo = vv
						end
					end
					if nodo["ID"]~= nil then
						red_resultante["neuronas"][k] = nodo
					end



				end
				if  red_resultante["neuronas"][k]["W"]~= nil and #red_resultante["neuronas"][k]["W"]>0 then -- si tenemos pesos los mutamos
					-- ahora mutamos los pesos
					-- print(red_resultante["neuronas"][k]["ID"])
					if mw== nil then
						mw = 0.4 -- fail safe , este valor no es obligatorio
					end
					--mutamos 1 solo peso, ya sea + o restando el peso
					for peso,vvv in pairs(red_resultante["neuronas"][k]["W"])  do
						if math.random(0,1000)/1000 < mr then
							local W = ((rng(1,1000)/1000)-(rng(1,1000)/1000)) * mw
							red_resultante["neuronas"][k]["W"][peso] = red_resultante["neuronas"][k]["W"][peso]+W
						end
					end
					---
					local candidatos = {}
					local capa = red_resultante["neuronas"][k]["layer"]
					--con esto vemos que peso podemos ++ a la neurona
					for k,v in pairs(red_resultante["neuronas"]) do
						if v["layer"]~= capa and ( (v["layer"] <capa) and v["layer"]~= -1)   then -- mantenemos una relacion linear

							table.insert(candidatos,v["ID"])

						elseif v["layer"]== -1 and  v["layer"]~= capa then

							table.insert(candidatos,v["ID"])
						end
					end
					-- ahora removemos de los candidatos las inovaciones nuevas que ya existan en nuestra neurona
					for k,v in pairs(red_resultante["neuronas"][k]["Inovations"]) do
						local match = false
						for o,vv in pairs(candidatos) do
							if v== vv then
								match = true
							end
						end
						if match== true then
							table.remove(candidatos,k)
						end

					end

					--
					if math.random(0,1000)/1000 < mr then -- si toca mutar mutamos
						if math.random(0,1) == 1 then -- si es = 1 agragamos un pesos
							local W = ((rng(1,1000)/1000)-(rng(1,1000)/1000)) * mw
							table.insert(red_resultante["neuronas"][k]["W"],W)
							local con= math.random(1,#candidatos)  --seleccionamos una Inovacion disponible
							table.insert(red_resultante["neuronas"][k]["Inovations"],candidatos[con] )
						else-- si no, quitamos uno
							if #red_resultante["neuronas"][k]["W"] == red_resultante["neuronas"][k]["Inovations"] then
								local peso_a_quitar = math.random(1,#red_resultante["neuronas"][k]["W"])
								table.remove(red_resultante["neuronas"][k]["W"][peso_a_quitar])
								table.remove(red_resultante["neuronas"][k]["Inovations"] [peso_a_quitar])
							end
						end
					end
					-- mutamos el bias
					if math.random(0,1000)/1000 < mr then
						local bias = ((rng(1,1000)/1000)-(rng(1,1000)/1000)) * mw
						red_resultante["neuronas"][k]["B"] = red_resultante["neuronas"][k]["B"] + bias
					end
				else -- si no los tenemos, hay una probavilidad de crearlos
					if math.random(0,1000)/1000 < mr then
						-------------------------------codigo para buscar nodos compatibles ---------------
						local mask = {}
						local candidatos = {}
						local capa = red_resultante["neuronas"][k]["layer"]
						for k,v in pairs(red_resultante["neuronas"]) do
							if v["layer"]~= capa and ( (v["layer"] <capa) and v["layer"]~= -1)   then -- mantenemos una relacion linear
								table.insert(mask,math.random(0,1))
								table.insert(candidatos,v["ID"])

							elseif v["layer"]== -1 and  v["layer"]~= capa then
								table.insert(mask,math.random(0,1))
								table.insert(candidatos,v["ID"])
							end
						end
						-- iteramos y generamos conexiones
						for candidato,valor in pairs(mask) do
							if valor == 1 then
								--    print("conexion de ",neurona["ID"] , " a ", candidatos[candidato],neurona["Inovations"] )
								table.insert(red_resultante["neuronas"][k]["Inovations"],candidatos[candidato])
								local W = ((rng(1,1000)/1000)-(rng(1,1000)/1000)) -- un peso para la conexion de -2 a 2
								table.insert(red_resultante["neuronas"][k]["W"],W)
								if  red_resultante["neuronas"][k]["B"] == 0 then
									local B = ((rng(1,1000)/1000)-(rng(1,1000)/1000)) -- un peso para el bias
									red_resultante["neuronas"][k]["B"] = B
								end
							end
						end

						-------------------------------codigo para buscar nodos compatibles ---------------
					end
				end
			end


		end


	end
	-- agragamos todos los nuevos nodos a la lista Nodos_Flatten
	for k,v in pairs(red_resultante["neuronas"]) do

		table.insert(red_resultante["Nodos_Flatten"],v["ID"])
	end
	return red_resultante,new_n
end

function NN.agregar_neurona_NEAT(capas_posibles,inovaciones,indice)
	local neurona = {"W","B","Inovations","act","ID","layer"}
	neurona["layer"]= math.random(1, capas_posibles)
	neurona["ID"] = "Neurona_"..tostring(indice)
	neurona["W"] = {}
	neurona["tipo"] = 2
	neurona["rnn"] = math.random(0,1)
	neurona["rnn_act"]=0
	neurona["rnn_Inovation"]={}
	-- neurona["Index"] = indice
	neurona["B"] = 0
	neurona["act"] = 0 -- la activacion de la neurona
	--  neurona["actfn"] =  0 -- la funcion activacion de la neurona que tambien evoluciona con la red
	if inovaciones ~= nil then
		neurona["Inovations"] = inovaciones -- coneciones con nodos
	else
		neurona["Inovations"] = {}
	end
	return neurona
end

function NN.crear_nn_NEAT(entradas,salidas,prob_nuevas_neuronas)-- tenemos x entradas y Z salidas
	local arquitectura = {}
	arquitectura["Nodos_Flatten"] = {} -- en esta tabla guardamos los nodos totales de la red para hacer los enlaces
	arquitectura["neuronas"] = {}
	arquitectura["capas"] = 1
	local indice = 1
	for i=1,entradas do
		local structura_neurona = {"act","ID","layer"}-- Neuro Evolution

		structura_neurona["layer"]= 0 -- entradas
		structura_neurona["ID"] = i -- id de la entrada
		--structura_neurona["W"] = {}

		--structura_neurona["B"] = {}
		structura_neurona["tipo"] = 0
		structura_neurona["rnn"] = math.random(0,1)
	    structura_neurona["rnn_act"]=0
		structura_neurona["rnn_act"]=0
		structura_neurona["rnn_Inovation"]={}
		structura_neurona["act"] = 0 -- la activacion de la neurona
		--structura_neurona["actfn"] =  0 -- la funcion activacion de la neurona que tambien evoluciona con la red
		structura_neurona["Inovations"] = {} -- coneciones con nodos
		table.insert(arquitectura["neuronas"],structura_neurona)
		table.insert(arquitectura["Nodos_Flatten"],tostring(i))
		indice = indice +1
	end
	for i=1,salidas do
		local structura_neurona = {"W","B","Inovations","act","ID","layer"}-- Neuro Evolution
		--  table.insert(arquitectura["neuronas"],structura_neurona)

		structura_neurona["layer"]= 99 -- salidas
		structura_neurona["ID"] = "out_"..tostring(i)
		structura_neurona["W"] = {}
		structura_neurona["tipo"] = 1
		structura_neurona["rnn"] = math.random(0,1)
	    structura_neurona["rnn_act"]=0
		structura_neurona["rnn_Inovation"]={}

		structura_neurona["B"] = 0
		structura_neurona["act"] = 0 -- la activacion de la neurona
		-- structura_neurona["actfn"] =  0 -- la funcion activacion de la neurona que tambien evoluciona con la red
		structura_neurona["Inovations"] = {} -- coneciones con no2dos
		table.insert(arquitectura["neuronas"],structura_neurona)
		table.insert(arquitectura["Nodos_Flatten"],"out"..tostring(i))
		indice = indice +1
	end
	---creamos coneciones aleratorias
	--primero por cada nerona creamos una mascara binaria {0,0,1,1,0,1} donde 1 es un enlace y 0 es nada
	--para todas las neuronas posibles compatibles (no sean de la misma capa)
	if math.random(0,100)/100 < prob_nuevas_neuronas then
		for i=1,math.random(1,3) do
			local num_actual= (#arquitectura["neuronas"] )- (entradas+salidas)
			local neurona =  NN.agregar_neurona_NEAT(1,nil,num_actual)
			table.insert(arquitectura["neuronas"],neurona)
			table.insert(arquitectura["Nodos_Flatten"],neurona["ID"])
		end

	end
	-- computamos las neuronas disponibles
	for n,neurona in pairs(arquitectura["neuronas"]) do
		local mask = {}
		local candidatos = {}
		local capa = neurona["layer"]
		for k,v in pairs(arquitectura["neuronas"]) do
			if v["layer"]~= capa and ( (v["layer"] <capa) and v["layer"]~= -1)  and math.random(0,1) then -- mantenemos una relacion linear
				table.insert(mask,math.random(0,1))
				table.insert(candidatos,v["ID"])

			elseif v["layer"]== -1 and  v["layer"]~= capa and math.random(0,1) then
				table.insert(mask,math.random(0,1))
				table.insert(candidatos,v["ID"])
			end
		end
		-- iteramos y generamos conexiones
		for candidato,valor in pairs(mask) do
			if valor == 1 then
				--    print("conexion de ",neurona["ID"] , " a ", candidatos[candidato],neurona["Inovations"] )
				table.insert(neurona["Inovations"],candidatos[candidato])
				local W = (((rng(1,1000)/1000)+(rng(1,1000)/1000))+((rng(1,1000)/1000)+(rng(1,1000)/1000)) ) -- un peso para la conexion de -4 a 4
				if rng(0,1) == 1 then W = W*-1 end
				table.insert(neurona["W"],W)
				if  neurona["B"] == 0 then
					local B = ((rng(1,1000)/1000)-(rng(1,1000)/1000)) -- un peso para el bias
					neurona["B"] = B
				end
			end
		end
	end
	return arquitectura
end
--predecir usando una arquitectura NEAT
function NN.predecir_NEAT(tabla_NEAT,x)
	local salidas = 0
	if tabla_NEAT == nil or x ==nil then
		print(tabla_NEAT,x)
		return nil
	end
	for k,v in pairs(tabla_NEAT["neuronas"]) do -- introducimos los valores a los nodos de entrada
		if v["layer"] == 99 then
			salidas = salidas+1
		end
		if v["layer"] == 0 then
			local id_entrada = v["ID"]
			if x[id_entrada] ~= nil then
				v["act"] = x[id_entrada]
			else
				print("La entrada no concuerda con los nodos de entrada")
			end
		end
	end
	-----------ahora predecimos a traves de los nodos ----
	local capa_actual = 0
	local nodos_ordenados = {} -- aqui ordenamos los nodos por capas para poder hacer una prediccion
	for k,v in pairs(tabla_NEAT["neuronas"]) do --
		if v["layer"] ~= 0 then -- no tomamos los nodos de entrada
			if v["layer"] ~= 99 then
				if nodos_ordenados[v["layer"]] == nil  then -- si en la lista nodos ordenados no existe ese indice de nodo , lo agregamos
					nodos_ordenados[v["layer"]] = {}
					table.insert(nodos_ordenados[v["layer"]] ,k)
					capa_actual = capa_actual+1
				end
			else -- si la neurona es una de salida, la agregamos al final con el indice mas alto
				if nodos_ordenados[capa_actual] == nil then
					nodos_ordenados[capa_actual] = {}
					table.insert(nodos_ordenados[capa_actual],k)--v["ID"] _old
				else
					table.insert(nodos_ordenados[capa_actual],k)
				end
			end
		end
	end
	-- ahora predecimos usando la tabla ordenada de neuronas
	local out = {}
	for k,capa in pairs(nodos_ordenados) do
		for k_c,neurona in pairs(capa) do
			local inovaciones = tabla_NEAT["neuronas"][neurona]["Inovations"]
			local valores = {}
			if tabla_NEAT["neuronas"][neurona]["rnn"] == 1 then
				if  #tabla_NEAT["neuronas"][neurona]["rnn_Inovation"] == 0 then
					-------------------------------------
					br = 0
					for k2,capa2 in pairs(nodos_ordenados) do
						for k_c2,neurona2 in pairs(capa2) do
								if math.random(0,1) == 1 then
									kkk=tabla_NEAT["neuronas"][neurona2]
									tabla_NEAT["neuronas"][neurona]["rnn_act"]= kkk["act"]
									tabla_NEAT["neuronas"][neurona]["rnn_Inovation"] = kkk["ID"]
								--	print(kkk)
								end
						end
						if br ==1 then break end
					end






					--------------------------------------------------------


				else
				--NN.tprint(tabla_NEAT["neuronas"][neurona]["rnn_Inovation"])
				rnn_con = tabla_NEAT["neuronas"][neurona]["rnn_Inovation"]
				for k2,capa2 in pairs(nodos_ordenados) do
						for k_c2,neurona2 in pairs(capa2) do
								if tabla_NEAT["neuronas"][neurona2]["ID"] == rnn_con then
									kkk=tabla_NEAT["neuronas"][neurona2]
									tabla_NEAT["neuronas"][neurona]["rnn_act"]= kkk["act"]

						--			print( kkk["act"])
								end
						end
						if br ==1 then break end
					end
					---------

				end
			end
			if #inovaciones ~= 0 then
				for o,v in pairs(tabla_NEAT["neuronas"]) do
					for i,inovation in pairs(inovaciones) do
						if inovation == v["ID"] then
							table.insert(valores,v["act"])
						end
					end
				end
				local pesos =  tabla_NEAT["neuronas"][neurona]["W"]
				local bias = tabla_NEAT["neuronas"][neurona]["B"]
				if tabla_NEAT["neuronas"][neurona]["rnn"] ==1   then
					table.insert(valores , tabla_NEAT["neuronas"][neurona]["rnn_act"])

				end
				if #valores~= #pesos   then
					local W = ((rng(1,1000)/1000)-(rng(1,1000)/1000))
					local diff = (#valores-#pesos)
					if diff >0 then --si los valores son mayores que los pesos
						for i=1,diff do
							table.insert(tabla_NEAT["neuronas"][neurona]["W"],W)
						end
					else
						diff = #pesos-#valores
						local tmppesos = {}
						for a,b in pairs(valores) do
							table.insert(tmppesos,tabla_NEAT["neuronas"][neurona]["W"][a])
						end
						tabla_NEAT["neuronas"][neurona]["W"]=tmppesos
					end
				end
			--	if tabla_NEAT["neuronas"][neurona]["rnn"] ==1  and  #tabla_NEAT["neuronas"][neurona]["W"] ~= #valores+1  then
				--	local W = ((rng(1,1000)/1000)-(rng(1,1000)/1000))
					--table.insert(tabla_NEAT["neuronas"][neurona]["W"],w)

			--	end
				if #valores == 0 then -- si no existe las neuronas de las inovaciones
					tabla_NEAT["neuronas"][neurona]["Inovations"]  = {}
					tabla_NEAT["neuronas"][neurona]["W"] = {}
					tabla_NEAT["neuronas"][neurona]["act"] =0
				else
					pesos =  tabla_NEAT["neuronas"][neurona]["W"]

					--  print(#valores,#pesos,bias,tabla_NEAT["neuronas"][neurona]["ID"],NN.dot(valores,pesos)) --DEBUG
					if pesos == nil or #pesos == 0 then
						print("pesos == nil ")
					end
					if valores== nil or #pesos == 0  then
						print("valores == nil")
					end
					if bias== nil or type(bias)=="table" then
						print("bias == nil")
					end

					local pred_linear = NN.dot(valores,pesos) + bias   -- predecirmos usando las inovaciones de las neuronas !!!
					--usamos la funcion de activacion
					if tabla_NEAT["neuronas"][neurona]["layer"] ~= 99 then
						if h_act == "tanh"  then
							pred_linear=tanh(pred_linear)
						elseif h_act =="sigmoid"  then
							pred_linear =NN.sigmoid(pred_linear)
						elseif h_act =="relu"  then
							pred_linear =NN.relu(pred_linear)
						elseif h_act =="leaky_relu"  then
							pred_linear =NN.leaky_relu(pred_linear,0.05)

						end
					else
						if out_actf == "tanh"  then
							pred_linear = tanh(pred_linear)
						elseif out_actf =="sigmoid"  then
							pred_linear =NN.sigmoid(pred_linear)
						elseif out_actf =="relu"  then
							pred_linear =NN.relu(pred_linear)
						elseif out_actf =="leaky_relu"  then
							pred_linear=NN.leaky_relu(pred_linear,0.05)
						end
					end
					tabla_NEAT["neuronas"][neurona]["act"] =pred_linear
					es_rnn=tabla_NEAT["neuronas"][neurona]["rnn"]

					if es_rnn == 1 then

						 tabla_NEAT["neuronas"][neurona]["rnn_act"]= pred_linear

					end
				end

				-- print( tabla_NEAT["neuronas"][neurona]["ID"]) --debug
				if tabla_NEAT["neuronas"][neurona]["layer"] == 99 then -- si es una predicion de una salida, la guardamos en una tabla
					table.insert(out, tabla_NEAT["neuronas"][neurona]["act"])
					--  print(NN.dot(valores,pesos)+bias,"SSSS" )
				end
			end
		end
	end
	if #out ~=salidas  and #out~= 0 and salidas~= 0 then
		local diff = salidas-#out

		for i=1,diff do
			table.insert(out,0)
		end
	elseif salidas~= 0 and #out ~=salidas  then
		for i=1,salidas do
			table.insert(out,0)
		end
	end
	return out
end

---------------------------Reinforcement learning---------------------------

function NN.fitness_medio()
	return 	NN.media(fitness)
end
local status = {} -- ver si ya acabamos de correr el batch en modo batch
function NN.iniciar_individuos(arch,individuos,porcentaje_elite,hyperparametro,tbatch,modo)
	fitness={}
	tba = tbatch
	modo_arch = modo
	--  print(tba,"SDDD")
	poblacion_agentes = {}
	poblacion_agentes_biases = {}
	global_arch = arch
	actual = 1
	for i=1,tbatch do
		table.insert(status,0)
	end
	batches_disponibles = math.ceil(individuos/tba)
	print("batches creados "..tostring(batches_disponibles))
	elite_indv = math.floor((individuos*porcentaje_elite)/100)
	if modo == "normal" then
		for i=1,individuos do
			local p,s,b= NN.crear_nn_normal(arch)

			table.insert(poblacion_agentes,p)
			table.insert(poblacion_agentes_biases,b)
			table.insert(fitness,0)
		end
		print("se ha creado  ",individuos," status: ok")
		init = true
		local p,s,b= NN.crear_nn_normal(arch)
		act_sheme = s
	elseif modo == "NEAT" then
		for i=1,individuos do
			--arch = {entradas,salidas prov_nuevas neuronas}
			local p= NN.crear_nn_NEAT(arch[1],arch[2],arch[3])

			table.insert(poblacion_agentes,p)
			--  print(#poblacion_agentes[#poblacion_agentes]["neuronas"])
			table.insert(fitness,0)
		end
		print("se ha creado  ",individuos," status: ok", #poblacion_agentes)
		init = true

	else
		print("No se especifico un modo ")

	end

	--print(act_sheme)
end


function NN.siguiente_individuo(log)
	if init == false then
		print("no se a iniciado el modo agente ml")
	else
		if actual < #poblacion_agentes then
			actual = actual+1
			if log ~= nil and log == true then
				print("individuo ",actual," ha sido seleccionado")
			end
		end
	end
end
---------------BATCH-----------------------
function NN.establecer_estado(individuo,estado)
	if status[individuo+1]~= nil then
		status[individuo+1] = estado
	else
		print("	Individuo > tbatch")
	end
end

function NN.termino_batch()
	local suma = 0
	for k,v in pairs(status) do
		if v == 0  then
			suma = suma+1
		end
	end
	if suma == 0  then
		return true
	else
		return false
	end
end

function NN.siguiente_batch()

	if tba*(batch_actual+1) < #poblacion_agentes then
		for k,v in pairs(status) do
			status[k]=0
		end
		batch_actual= batch_actual+1
		actual = tba*(batch_actual)
		print("batch ",batch_actual+1," ha sido seleccionado")
	else
		print("no mas individuos disponibles")
	end

end

function NN.establecer_recompenza_batch(valor,indice_batch)
	if fitness[actual+indice_batch] == nil then
		print("el individuo no existe")
	end
	if valor == "ayuda" then
		print("los valores tomaran el mayor fitness como el mejor")

	else
		fitness[actual+indice_batch]  = fitness[actual+indice_batch]+valor
	end

end

function NN.predecir_individuo_batch(individuo,x,rnn)
	if modo_arch == "normal"then
		return NN.predecir(x,poblacion_agentes[actual+individuo],poblacion_agentes_biases[actual],act_sheme,rnn)
	elseif modo_arch== "NEAT" then
		return NN.predecir_NEAT(poblacion_agentes[actual+individuo],x)
	end

end
----------------------------SOLO UNO-------------------
function NN.predecir_individuo(x,rnn)
	if modo_arch == "normal"then
		return NN.predecir(x,poblacion_agentes[actual],poblacion_agentes_biases[actual],act_sheme,rnn)
	elseif modo_arch== "NEAT" then
		return NN.predecir_NEAT(poblacion_agentes[actual],x)
	end


end




function NN.predecir_mejor_individuo(x,rnn)
	if modo_arch == "normal"then
		return NN.predecir(x,mejor_agente,mejor_agente_biases,act_sheme,rnn)
		--    return NN.predecir(x,poblacion_agentes[actual],poblacion_agentes_biases[actual],act_sheme,rnn)
	elseif modo_arch== "NEAT" then
		--  return NN.predecir(x,mejor_agente,mejor_agente_biases,act_sheme,rnn)
		-- print(mejor_agente,x)
		return NN.predecir_NEAT(mejor_agente,x)
	end



end
--------------------------------------------------------
local function quantize (a,range)

	a = math.ceil( 128*(a/range))

return a
end

local function get_arr_range(x)
	maximum = -math.huge
	minimum = math.huge
	for kk,vv in pairs(x) do
		if vv > maximum then
			maximum = vv
		end
		if vv <minimum then
			minimum = vv
		end



	end

	if minimum < 0 then
		if -minimum > maximum then
			return minimum
		end

	end

	if maximum < 0 then
		maximum = minimum
	end
	return maximum
end

local function append(x,y)
	local result = {}
	for kk,vv in pairs(y) do
		table.insert(result,vv)

	end
	for kk,vv in pairs(x) do
		table.insert(result,vv)

	end


return result
end


function NN.print_mejor()
	if modo_arch == "normal" then
		if io ~= nil then
			for k,v in pairs(mejor_agente ) do
				io.write("W ")
				io.write("{")
				for k1,v1 in pairs(v) do

					io.write("{")
					for k2,v2 in pairs(v1) do
						io.write(v2)
						if(k2 ~= #v1 ) then
							io.write(" , ")
						end

					end
					io.write("}")
					if(k1 ~= #v) then
						io.write(" , ")
					end

				end
				io.write("}")
				io.write("  ")
				io.write("b {")
				for k1,v1 in pairs(v) do
					io.write(mejor_agente_biases[k][k1] )
					if(k1 ~= #v ) then
						io.write(" , ")
					end
				end
				io.write("}")
				io.write("\n")
				--------------------------------
				io.write("quantized:" )
				io.write("W ")
				io.write("{")
				for k1,v1 in pairs(v) do
					rr = get_arr_range(append(v1 ,mejor_agente_biases[k] ) )
					io.write("{")
					for k2,v2 in pairs(v1) do
						io.write(quantize(v2,rr))
						if(k2 ~= #v1 ) then
							io.write(" , ")
						end

					end
					io.write("}")
					if(k1 ~= #v) then
						io.write(" , ")
					end

				end
				io.write("}")
				io.write("  ")
				io.write("b {")

				for k1,v1 in pairs(v) do
					io.write(quantize(mejor_agente_biases[k][k1],rr ))
					if(k1 ~= #v ) then
						io.write(" , ")
					end
				end
				io.write("}")
				io.write(" dequantition val ".. tostring(rr) )
				io.write("\n")
				io.write("\n")

			end
		else

			if game:GetService("HttpService")~= nil then
				local HttpService = game:GetService("HttpService")
				local json = HttpService:JSONEncode(mejor_agente)
				print("W ",json)
				local json = HttpService:JSONEncode(mejor_agente_biases)
				print("-----------------------------------")
				print("b ",json)
			end
		end
	else
		NN.tprint(mejor_agente)

	end
end

function NN.establecer_recompenza(valor)

	if valor == "ayuda" then
		print("los valores tomaran el mayor fitness como el mejor")

	else
		fitness[actual] = valor
	end

end


function NN.leer_fitness()
	return fitness[actual]
end
local wd = 0 -- para no causar lag, cada 10 gen podemos llamar un print

local media_historica = 0
local m_WD = 0

function NN.crear_nueva_generacion(pft)
	local hyperparametro = config["mr"]
	local mWS = config["Wms"]
	local addrt = config["addrt"]

	--config["wd"]= 3
	--config["mr"] = 0.9
	--config["Wms"]= 1
	--config["addrt"] = 0.3
	--config["crossrt"] = 0.5 -- 1/2
	if  m_WD ~= config["wd"] then
		media_historica = media_historica + NN.media(fitness)
		if m_WD == 0 then
			print("Fitness", media_historica)
		else
			print("Fitness",media_historica/m_WD )
		end

	elseif   m_WD == config["wd"] then
		m_WD = 0
		media_historica = media_historica/ config["wd"]
		print("WD_Service ",media_historica)
		if media_historica > prev_wd_fitness then
			prev_wd_fitness = media_historica
		else
			config["Wms"] = config["Wms"] *0.99
			print("WD_Service new wms",config["Wms"] )
		end

	end

	m_WD = m_WD+1

	local tablaord,indices =NN.ordenar(fitness)
	if pft ~= nil and pft == true and wd >= 10 then --pft es print fitness, el promedio del fitness de la generacion
		print("Fitness medio "..tostring(NN.media(fitness)))

	end
	wd = wd+1
	local tmppob = {}
	local tmpbias = {}
	local bf = -math.huge
	local bi = 0
	batch_actual = 0

	for k,v in pairs(status) do
		status[k]=0--reset watchdog
	end
	for k,v in pairs(fitness) do
		if v > bf and v~= 0 then
			bf = v
			bi = k
		end
		--fitness[k] = 0
	end
	if bf > mejor_fitness_global and bf~= 0 then
		mejor_fitness_global = bf

		if modo_arch == "normal" then
			mejor_agente = NN.deepCopy(poblacion_agentes[bi])
			mejor_agente_biases =NN.deepCopy(poblacion_agentes_biases[bi])
		elseif  modo_arch =="NEAT" then
			mejor_agente = poblacion_agentes[bi]
		end
		print("mejor individuo, fitness ",bf,poblacion_agentes[bi])
	end

	for ia=0,elite_indv do
		if indices[ia]~= nil then
			table.insert(tmppob,poblacion_agentes[indices[ia]])
			--                print(" 1428",#tmppob[#tmppob]["neuronas"])
			if modo_arch == "normal" then
				table.insert(tmpbias,poblacion_agentes_biases[indices[ia]])
			end
		end
	end
	for o = 0,5 do
		if modo_arch == "normal" then
			local redtt,_,bb= NN.crear_nn_normal(global_arch)
			table.insert(tmppob,redtt)
			table.insert(tmpbias,bb)
		elseif modo_arch == "NEAT" then
			local redtt,added = NN.crear_nn_NEAT(global_arch[1],global_arch[2],config["addrt"])

			table.insert(tmppob,redtt)
		end
	end
	--DEBUG NN.tprint(tmppob)
	if modo_arch == "normal" then
		--      print("N")
		poblacion_agentes,poblacion_agentes_biases = NN.crear_desendientes(tmppob,tmpbias,#poblacion_agentes,hyperparametro,addrt,mWS)
		-- print(#poblacion_agentes,#poblacion_agentes_biases)
	elseif modo_arch == "NEAT" then
		--     print("NEAT")
		poblacion_agentes = NN.crear_desendientes(tmppob,nil,#poblacion_agentes,hyperparametro,addrt,mWS)
		--    print(#poblacion_agentes)
	end
	for k,v in pairs(fitness) do

		fitness[k] = 0
	end
end
--------------------optimizador genetico para datos ------------------------
function NN.optimizador_genetico_con_datasets(x,y,arch,ind,criterio,rnn,porcentaje,lua_se_bloquea_sin_wait,minimo_hyperparametro,hyperparametro)
	local red,act= NN.crear_nn_normal(arch)
	local poblacion={}
	local poblacionbias={}
	local mejor={}
	local mejorb={}
	local mejorl=math.huge
	local elite= math.floor((ind*porcentaje)/100)
	local prediciones1={}
	for i=0,ind do
		local p,_,b= NN.crear_nn_normal(arch)
		table.insert(poblacion,p)
		table.insert(poblacionbias,b)
	end
	print(#poblacion,elite)
	local tmp
	local tmppob
	local tmpbias
	local prediciones={}
	local printinterval = 10
	local wathcdog = 0

	local inicio = os.clock()

	local logger = 0
	local paso = false

	while  mejorl ~=0 and  mejorl > criterio   and hyperparametro >minimo_hyperparametro do

		local errores={}
		tmppob={}
		tmpbias = {}
		if wathcdog == printinterval then
			wathcdog = 0
			--io.write("#")
			if lua_se_bloquea_sin_wait then
				wait()

			end
			if not paso then
				hyperparametro = hyperparametro - hyperparametro*0.01
				--print("decay ",hyperparametro)
			end
			paso = false
		end
		wathcdog = wathcdog+1
		for i=1,ind do
			prediciones={}
			for k,v in pairs(x) do
				table.insert(prediciones,NN.predecir(x[k],poblacion[i],poblacionbias[i],act,rnn))


			end

			tmp=NN.mse(y,prediciones)
			if tmp< mejorl then
				mejorl=tmp
				paso = true
				mejorb=NN.deepCopy(poblacionbias[i])
				mejor=NN.deepCopy(poblacion[i])
				print("se encontro mejor",tmp,hyperparametro)
				prediciones1 = {}
				logger = logger+1

			end
			table.insert(errores,tmp)

		end
		local tablaord,indices =NN.ordenar(errores)
		for ia=0,elite do
			table.insert(tmppob,poblacion[indices[#poblacion-ia]])
			table.insert(tmpbias,poblacionbias[indices[#poblacionbias-ia]])
		end
		for o = 0,5 do
			local redtt,_,bb= NN.crear_nn_normal(arch)
			table.insert(tmppob,redtt)
			table.insert(tmpbias,bb)
		end
		poblacion,poblacionbias = NN.crear_desendientes(tmppob,tmpbias,#poblacion,hyperparametro)

	end
	prediciones1={}
	for k,v in pairs(x) do
		local kkk=NN.deepCopy( NN.predecir(v,mejor,mejorb,act,rnn) )
		table.insert(prediciones1,kkk)
	end
	tmp=NN.mse(y,prediciones1)
	print("tardo : ",(os.clock()-inicio)/60)
	print('mejor',tmp,mejorl)
	for k,v in pairs(prediciones1) do
		for kkk,vvv in pairs(v) do io.write(" ",vvv) end
		for k31,v31 in pairs(y[k]) do io.write(" ",v31) end
		io.write("\n")
	end
	print("----------------------------------")
	mejor_agente = mejor
	mejor_agente_biases = mejorb

	NN.print_mejor()


	return 	mejor,mejorb,act


end

return NN
