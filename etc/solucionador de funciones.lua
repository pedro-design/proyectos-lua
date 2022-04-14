
local function mse(x,y) -- el error medio cuadrado
 if x==nil or y==nil then
    return 9999
 end

	if type( x ) ~= "table"  and type( y ) ~= "table" then
		return math.sqrt((x-y)*(x-y))
	elseif type(x)=="table" and type(y)=="table" then
		local sum = 0
		for k,v in pairs(x) do
			sum = sum+ math.sqrt((x[k]-y[k])*(x[k]-y[k]))


		end
		return sum
	end
end

--bloques matematicos
local function al_cuadrado(x)
	return x*x
end

local function al_cubo(x)
	return x*x*x
end

local function seno(x)
	return math.sin(x)
end

local function coseno(x)
return math.cos(x)
end

local function tangente(x)
return math.tan(x)
end


local function raiz_cuadrada(x)
return math.sqrt(x)
end


local pi=3.1415926535
------------------------------

math.randomseed(os.time())
local rng = math.random
---------------------------------
local iteraciones =1500
local variables = {"h","b"}
local tam=#variables
local maxsol= 30-- maximo de soluciones a buscar
local solo_enteros = false
local ppp=200
local soloex=false  --solo soluciones exactas o tambien aproximaciones
local bueno
if soloex then bueno=0 else bueno=0.001 end
local valor_a_buscar =100-- edita este valor,de acuerto al valor que buscas
local function tu_ecuacion(genoma)--no resuelve sistemas de ecuaciones, solo lineales
 local cuenta=0
 for k,v in pairs(genoma) do
    cuenta=cuenta+1

 end
if cuenta==tam then-- checksum de integridad
	return genoma["h"]*genoma["b"] -- edita esto con tu problema
	-- se traduce a altura*base
end

end

local function recursive_compare(t1,t2)
  if t1==t2 then return true end
  if (type(t1)~="table") then return false end

  local mt1 = getmetatable(t1)
  local mt2 = getmetatable(t2)
  if( not recursive_compare(mt1,mt2) ) then return false end


  for k1,v1 in pairs(t1) do
    local v2 = t2[k1]
    if( not recursive_compare(v1,v2) ) then return false end
  end
  for k2,v2 in pairs(t2) do
    local v1 = t1[k2]
    if( not recursive_compare(v1,v2) ) then return false end
  end

  return true
end


local function deepCopy(original)-- con esto clonamos las tablas
    	local copy = {}
    	for k, v in pairs(original) do
    		if type(v) == "table" then
    			v = deepCopy(v)
    		end
    		copy[k] = v
    	end
    	return copy
end


local function genetic_poblation(args,size) -- con esto creamos la poblacion
	local poblation = {}
	for i=0,size,1 do
		poblation[i] = {}
		for _,v in pairs(args) do
			if rng(0,1) == 1 then
				poblation[i][v]= -rng(0,100)
			else
				poblation[i][v]= rng(0,100)
			end

		end
	end
	return poblation
end

local function cross_over(tb1,tb2,ops,mult,ints) -- con este se cruzan los genes
	local genome1 = {}
	local genome2 = {}
	for k,v in pairs(tb1) do
		if ops=="|" and not ints then
			if rng(0,1) == 1 then
				genome1[k]=tb2[k]+ ( (rng(0,1000)/1000) -(rng(0,1000)/1000)   )*mult
				genome2[k]=tb1[k]+ ( (rng(0,1000)/1000) -(rng(0,1000)/1000)   )*mult
			else

				genome1[k]=tb2[k]- ( (rng(0,1000)/1000) -(rng(0,1000)/1000)   )*mult
				genome2[k]=tb1[k]- ( (rng(0,1000)/1000) -(rng(0,1000)/1000)   )*mult
			end

		elseif ops=="+" and not ints    then
			genome1[k]=tb2[k]+ ( (rng(0,1000)/1000))*mult
			genome2[k]=tb1[k]+ ( (rng(0,1000)/1000))*mult
		elseif ops=="-" and not ints   then
			genome1[k]=tb2[k]- ( (rng(0,1000)/1000))*mult
			genome2[k]=tb1[k]- ( (rng(0,1000)/1000))*mult
		end

		if ops=="|" and  ints then
			if rng(0,1) == 1 then
				genome1[k]=tb2[k]+ ( (rng(0,10) -(rng(0,10)  )))
				genome2[k]=tb1[k]+ ( (rng(0,10)) -(rng(0,10)   ))
			else
				genome1[k]=tb2[k]- ( (rng(0,10) -(rng(0,10)  )))
				genome2[k]=tb1[k]- ( (rng(0,10)) -(rng(0,10)   ))

			end

		elseif ops=="+"  and ints    then
			genome1[k]=tb2[k]+ ( (rng(0,10) -(rng(0,10)   )))
			genome2[k]=tb1[k]+ ( (rng(0,10)) -(rng(0,10)   ))
		elseif ops=="-"  and  ints   then
			genome1[k]=tb2[k]- ( (rng(0,10) -(rng(0,10)  )))
			genome2[k]=tb1[k]- ( (rng(0,10)) -(rng(0,10)   ))
		end

	end
	return genome1,genome2

end

local function r3(x,y)
   return ( x*y)/50

end
local criterio=math.floor( (20*ppp) /100 )
print("tomando en cuenta solo",criterio)


local function ordenar(x)
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





local function evolve(poblation,target_val)-- con este evolucionamos los valores
	local size = #poblation
	local values ={}
	local tmp_score = {}
	local best
	local best_table = {}
	local best_index = 0
	local soluciones = {}
	local costos = {}
	local multiplicador =1
 local mean=0
 local tmp_poblation={}
	for i=0,iteraciones,1 do-- busqueda nomal
		tmp_score = {}
		for k, v in pairs(poblation) do
			tmp_score[k] = mse(tu_ecuacion(poblation[k]),target_val)


			values[k] = tu_ecuacion(poblation[k])


		end
		local tmp = deepCopy(tmp_score)
		table.sort(tmp_score, function(a,b) return a > b end)
		best = tmp_score[#tmp_score]
		for k,v in pairs(tmp) do
			if v == best then
				best_table = deepCopy(poblation[k])
				best_index = k

			end
		end
		if  mse(tu_ecuacion(best_table),target_val) <0.1 then
				multiplicador =  mse(tu_ecuacion(best_table),target_val)

			else
				multiplicador = 1
		end
  if mse(tu_ecuacion(best_table),target_val) ==0 then
    print('solucion exacta encontrada en ',i ,'iteraciones')
    break
  end

		for k, v in pairs(poblation) do
			if rng(0,1) == 1 then
				local p1,p2= rng(1,size),rng(1,size)
				while p1== best_index or p2 == best_index do
					p1,p2= rng(1,size),rng(1,size)
				end
				poblation[p1],poblation[p2] =  cross_over(poblation[p1],poblation[p2],"|",multiplicador,solo_enteros)
			else
				local p1,p2= rng(1,size),rng(1,size)
				while p1== best_index or p2 == best_index do
					p1,p2= rng(1,size),rng(1,size)
				end
				poblation[p1],poblation[p2] =  cross_over(poblation[p1],poblation[best_index],"|",multiplicador,solo_enteros)
			end

		end
		for k,v in pairs(poblation[rng(1,size)]) do
			if rng(0,1) == 1 then
				v= -rng(1,100)
			else
				v= rng(1,100)
			end

		end
	end

	table.insert(soluciones,best_table)

	table.insert(costos,best)
print('iniciando busqueda recursiva')
	best_table = {}
	local positivo= true
	for k4,index in pairs( poblation) do
		for k5,value in pairs( index) do
			poblation[k4][k5] = rng(0,100)-rng(0,100)--el lado neutral
		end
	end
	local ended = false
	tmp_score = {}
 local i=0
	 while i<iteraciones and #soluciones <maxsol+2 do-- busqueda del lado + y - de la solucion neutral
		tmp_score = {}
  mean=0
  tmp_poblation={}
		for k, v in pairs(poblation) do
			tmp_score[k] = mse(tu_ecuacion(poblation[k]),target_val)
			values[k] = tu_ecuacion(poblation[k])
			if  mse(tu_ecuacion(poblation[k]),target_val) <1 then
				multiplicador = mse(tu_ecuacion(poblation[k]),target_val)
			else
				multiplicador = 1
			end

		end
		local tmp = deepCopy(tmp_score)
		table.sort(tmp_score, function(a,b) return a > b end)
		best = tmp_score[#tmp_score]
		for k,v in pairs(tmp) do
			for k1,solucion in pairs(soluciones) do
				if v == best and not recursive_compare(poblation[k],soluciones[k1]) then
					 if mse(soluciones[k1],poblation[k]) > 0.01 then
					best_table = deepCopy(poblation[k])
					best_index = k
				elseif solo_enteros and recursive_compare(poblation[k],soluciones[k1]) then
						for k4,index in pairs( poblation) do
							for k5,value in pairs( index) do
								poblation[k4][k5] = rng(0,100)-rng(0,100)--puro rng
							end
						end
				 	end
				end
			end
		end
		if  mse(tu_ecuacion(best_table),target_val) <0.1  then
				multiplicador =mse(tu_ecuacion(best_table),target_val)
			else
				multiplicador = 1
		end

	if mse(tu_ecuacion(best_table),target_val) <=bueno  then
		 local found = false
   local mse1= mse(tu_ecuacion(best_table),target_val)
  local mse2
		 for k,v in pairs(soluciones) do
   mse2=mse(tu_ecuacion(soluciones[k]),target_val)
			if mse(best_table,soluciones[k]) <1 and mse1>mse2  then
			   found=true
  elseif mse(best_table,soluciones[k]) <1 and   mse1<mse2 then
     soluciones[k]=deepCopy(best_table)
    -- print("mejor valor en")
     found=true
  elseif mse(best_table,soluciones[k]) <1 then
      found=true
			end
		 end

		 if not found then
			table.insert(soluciones,best_table)
			table.insert(costos,best)
			print('solucion nueva encontrada en',i,' iteraciones')
			i=0
			for k4,index in pairs( poblation) do
				for k5,value in pairs( index) do
					poblation[k4][k5] = rng(0,100)-rng(0,100)--puro rng
				end
			end
		end
	end
local ordenados,indicesord= ordenar(tmp)
for i=0,criterio do
 table.insert(tmp_poblation,poblation[indicesord[#indicesord-i]])

end
while #tmp_poblation ~=#poblation do
   local pp=#tmp_poblation
   while pp<2 do
      table.insert(tmp_poblation,best_table)
      pp=#tmp_poblation
    end
   local x1,x2= rng(1,pp),rng(1,pp)
   local tmp,_=cross_over(tmp_poblation[x1],tmp_poblation[x2],"|",multiplicador,solo_enteros)
   table.insert(tmp_poblation,tmp)

end
poblation=deepCopy(tmp_poblation)
positivo=rng(0,1)
i=i+1
	for k, v in pairs(poblation) do
		if rng(0,1)==1 then
		for k,v in pairs(poblation[rng(1,size)]) do
			if rng(0,1) == 1 and positivo== 1 then
				v= rng(1,100)

			elseif  rng(0,1) == 1 and positivo== 0 then
				v=- rng(1,100)
			end

		end
end
end
end
--table.insert(soluciones,best_table)
--table.insert(costos,best)
return costos,soluciones


end



local p = genetic_poblation(variables,ppp)--esta es la poblacion, de 100 individuos
local costos,soluciones = evolve(p,valor_a_buscar)--busca una solucion cercana
print("recuerda editar tu ecuacion en este programa")
for k,v in pairs(soluciones) do
	print("solucion",k,"mse:",costos[k],"valores: ")
	for k1,v1 in pairs(v) do
		print(k1,v1)-- devuelve los valores optimos de 1 posible solucion
	end

		--print(k,v)-- devuelve los valores optimos de 1 posible solucion
end
print("toma en cuenta solo las soluciones con un mse cercano a 0")
print("presiona enter para cerrar")
print(os.clock())

local f = io.read()
