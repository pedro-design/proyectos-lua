

local NN = {}
local h_act = 0
local out_actf = 0
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

local huge = math.huge
local rng = math.random
local exp = math.exp
local sqrt= math.sqrt
local tanh = math.tanh
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
		for k=1,#original  do
			local v = original  [k];
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
				tmp=tmp + ((x[k][k1]-y[k][k1])^2)
			end
		end
		return tmp/#x

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
					local w =(NN.noise(10)+NN.noise(10)+NN.noise(10))--*math.sqrt(2 /capas[k-1]+capas[k])
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
					local w =(NN.noise(10)+NN.noise(10)+NN.noise(10))  -- *math.sqrt(2 /capas[k-1]+capas[k])
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


function string:split (sep)
    local sep, fields = sep or ":", {}
    local pattern = string.format("([^%s]+)", sep)
    self:gsub(pattern, function(c) fields[#fields+1] = c end)
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
	activaciones = NN.deepCopy(activaciones1) -- problemas de memoria
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
		return activaciones[#activaciones]


	else--conectamos la ultima salida a la entrada actual, modo rnn

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


function NN.crear_desendientes(tmppob,tmpbias,tamaño_de_poblacion,hyperparametro)
	local tmppob = NN.deepCopy(tmppob)
	local tmpbias = NN.deepCopy(tmpbias)
	--print(#tmppob,#tmpbias,tamaño_de_poblacion)
	while #tmppob ~=tamaño_de_poblacion do
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


end


--------------------optimizador genetico para datos ------------------------
function NN.optimizador_genetico_con_datasets(x,y,arch,ind,criterio,rnn,porcentaje,lua_se_bloquea_sin_wait,minimo_hyperparametro,hyperparametro)
	local red,act= NN.crear_nn(arch)
	local poblacion={}
 local poblacionbias={}
	local mejor={}
 local mejorb={}
	local mejorl=math.huge
	local elite= math.floor((ind*porcentaje)/100)

	for i=0,ind do
		local p,_,b= NN.crear_nn(arch)
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

	  errores={}
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
		local redtt,_,bb= NN.crear_nn(arch)
		table.insert(tmppob,redtt)
		table.insert(tmpbias,bb)
	  end
	poblacion,poblacionbias = NN.crear_desendientes(tmppob,tmpbias,#poblacion,hyperparametro)

	end
	local prediciones1={}
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
	for k,v in pairs(mejor ) do
		for k1,v1 in pairs(v) do

			for k2,v2 in pairs(v1) do
				io.write(v2)
				io.write(" ")

			end
  io.write(" b ",mejorb[k][k1])
			io.write("\n")
		end

	end


	return 	mejor,mejorb,act


end

return NN



---
