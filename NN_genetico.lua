local rng = math.random
local sqrt= math.sqrt

local NN = {}
local h_act = 0
local out_actf = 0
function NN.r3(x,y)
	return (x*y)/50

end

function NN.media(x)
	local suma = 0
	for _,v in pairs(x) do
		suma = suma+v
	end
	return suma/ #x
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


function NN.sigmoid(x)
		return 1 / (1 + math.exp(-x))
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

function NN.RNG(x,y)

	return rng(x,y)
end
function NN.out_act(x)
	out_actf = x
end
function NN.hact(x)
	h_act = x
end


function NN.cross_over_1d(x,y,m)
   local x1 = {}
   local y1 = {}
   for k,v in pairs(x) do
		local n = NN.noise(10)
       if NN.RNG(0,100)/100 < m then
			table.insert(x1,x[k])
			table.insert(y1,y[k])

	   else
			table.insert(y1,x[k])
			table.insert(x1,y[k])
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
		for k,v in pairs(y) do
			for k1,v1 in pairs(y[k]) do
				tmp=tmp + ((x[k][k1]-y[k][k1])^2)
			end
		end
		return tmp
end

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
	elseif type(x)== 'table' and type(y) ~= 'table'  then
		for i=1,#x,1 do
			sum=sum+ (x[i]*y)
		end
	end
	return sum
end

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
	local maximo=-math.huge
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
		maximo=-math.huge
		for k,v in pairs(x) do
			if v>=maximo and v<ordenados[#ordenados] then
				maximo=v
				tmpindice=k
			end
		end
		if maximo ~= -math.huge then
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

function NN.crear_nn(capas)
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
					local w =NN.noise(10)  --*math.sqrt(2 /capas[k-1]+capas[k])
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
					local w =NN.noise(10) *math.sqrt(2 /capas[k-1]+capas[k])
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





function NN.predecir(x,pesos,activaciones)
	local indice = 1

	activaciones = NN.deepCopy(activaciones) -- problemas de memoria
	if #x ~= #pesos[1] then print("tamaños no concuerdan,",#x,#pesos[1]) end
	for k,v in pairs(pesos) do
		if k ==1  then

			for k1,v1 in pairs(v) do
				if h_act == "tanh"  then
					activaciones[k][k1] =math.tanh(NN.dot(x,v1))
				elseif h_act =="sigmoid"  then
					activaciones[k][k1] =NN.sigmoid(NN.dot(x,v1))
				elseif h_act =="relu"  then
					activaciones[k][k1] =NN.relu(NN.dot(x,v1))
				elseif h_act =="leaky_relu"  then
					activaciones[k][k1] =NN.leaky_relu(NN.dot(x,v1),0.05)

				else
					activaciones[k][k1] =(NN.dot(x,v1))
				end
			end
			indice = indice+1
		elseif k~= #activaciones then
			for k1,v1 in pairs(v) do
				if h_act == "tanh"  then
					activaciones[k][k1] = math.tanh(NN.dot(activaciones[k-1],v1))
				elseif h_act =="sigmoid"   then
					activaciones[k][k1] =NN.sigmoid(NN.dot(activaciones[k-1],v1))
				elseif h_act =="relu"  then
					activaciones[k][k1] =NN.relu(NN.dot(activaciones[k-1],v1))
				elseif h_act =="leaky_relu"  then
					activaciones[k][k1] =NN.leaky_relu(NN.dot(activaciones[k-1],v1),0.05)
				else
					activaciones[k][k1] =(NN.dot(activaciones[k-1],v1)) --linear no sirve
				end
			end
		else
			for k1,v1 in pairs(v) do
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
			end
		--	print("fin")
		end
	--	indice = indice+1
	end

	return activaciones[#activaciones]
end

function NN.noise(x)
	local sum = 0
	for i = 1,x do
		sum = sum + (((rng(0,1000)/1000)-0.5)/0.5 )/i
	end
	return sum
end


function NN.mutar(x,y,z,grad,res)
	local tmp=NN.deepCopy(x)
	local tn = 0
	for k,v in pairs(x) do
		for k1,v1 in pairs(v) do
			for k2,v2 in pairs(v1) do
			---	tn = 0
			--	if rng(0,1000)/1000 <cr then
					--tn= NN.noise(res)

					tmp[k][k1][k2]=v2+ (( y[k][k1][k2] - z[k][k1][k2])*grad)+tn*0.001

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


return NN



---
