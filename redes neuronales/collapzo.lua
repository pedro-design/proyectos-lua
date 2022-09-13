local wavecollapse= {}
local array = {}
local ladrillos = {}
local x_s = 100
local y_s = 100
local indefinidos = {}




function read_array(x,y,array,write,val)
	if write== true	 then
		array[x+(y*y_s)] = val
	end
	return array[x+(y*y_s)]
end


local configs = {}
local indices = {}
function wavecollapse.set_size(x,y)
	 x_s = x
	 array = {}
	 indefinidos = {}
 y_s = y
	for i=1,x_s do
	for ii=1,y_s do
		table.insert(array,0)
		table.insert(indefinidos,{i,ii})
	end
end
end

function wavecollapse.add_rules(configss)
configs = configss
for k,v in pairs(configs) do
	table.insert(indices,k)
	--print(k)
end
end

function parse(estilo,parte)
--	print(estilo)
	--parte.Color = estilo["color"]
end
local current_x=math.random(1,x_s-1)
local current_y = math.random(1,y_s-1)
local s=indices[math.random(1,#indices)]





read_array(current_x,current_y,array,true,s) -- escribimos el estado actual de la celda



entropia = 60





local function index_to_coords(index,size)
	y = math.ceil(size/ index)
	x = index-y
	return x,y


end

 function wavecollapse.check_collapse(x,y,arr)
	local current_tittle = read_array(x,y,arr,false,nil)
	local find = false

	if current_tittle== 0 then
		local s=indices[math.random(1,#indices)]
		parse(configs[s],read_array(x,y,ladrillos,false,nil))

		read_array(x,y,array,true,s)
		current_tittle = read_array(x,y,arr,false,nil)
	--	local modelo = configs[current_tittle]["modelo"]:Clone()
		--modelo.Parent = game.Workspace
		local current_tittle_p = read_array(x,y,ladrillos,false,nil)

	--	modelo:SetPrimaryPartCFrame(CFrame.new(current_tittle_p.Position)*modelo.Floor.CFrame.Rotation)
		--current_tittle_p:Destroy()
	end
	if (x > 1 and x<(x_s)) and (y > 1 and y<(y_s) and current_tittle~= 0) then
		--print( configs[current_tittle]["modelo"])
	--	local modelo = configs[current_tittle]["modelo"]:Clone()
	--	modelo.Parent = game.Workspace
		local current_tittle_p = read_array(x,y,ladrillos,false,nil)

	--	modelo:SetPrimaryPartCFrame(CFrame.new(current_tittle_p.Position))
			--print(x,y,current_tittle)
		--current_tittle_p:Destroy()
		local posible_k = configs[current_tittle]["vecinos"]

		--		print("Valid convolution")
		local up =  read_array(x,y+1,arr,false,nil)
		local down =  read_array(x,y-1,arr,false,nil)
		local left =  read_array(x-1,y,arr,false,nil)
		local right =  read_array(x+1,y,arr,false,nil)
		local superposiciones = {}

		-- buscamos celdas vacias o indefinidas
		if up ==0  then
			local total_probs = {}
			local final = {}
			local x_q = x
			local y_q = y+1
			local sum = 0
			local chose_tittle = read_array(x_q,y_q,arr,false,nil)
			--*	local pprobs = {}
			local probs =  {read_array(x_q,y_q+1,arr,false,nil),
				read_array(x_q,y_q-1,arr,false,nil),
				read_array(x_q-1,y_q,arr,false,nil),
				read_array(x_q+1,y_q,arr,false,nil)}
			for k,v in pairs(probs) do


				if v~= 0  and v ~= nil then
					for kk,vv in pairs(configs[v]["vecinos"]) do

						sum = sum+1
						if total_probs[vv] == nil then
							total_probs[vv]=0
						end
						total_probs[vv] = total_probs[vv]+1
					end
				end
			end

			superposiciones[1] = total_probs
		else
			superposiciones[1] = 0
		end

		if down ==0  then
			local total_probs = {}
			local final = {}
			local x_q = x
			local y_q = y+1
			local sum = 0
			local chose_tittle = read_array(x_q,y_q,arr,false,nil)
			--*local pprobs = {}
			local probs =  {read_array(x_q,y_q+1,arr,false,nil),
				read_array(x_q,y_q-1,arr,false,nil),
				read_array(x_q-1,y_q,arr,false,nil),
				read_array(x_q+1,y_q,arr,false,nil)}
			for k,v in pairs(probs) do


				if v~= 0  and v ~= nil then
					for kk,vv in pairs(configs[v]["vecinos"]) do

						sum = sum+1
						if total_probs[vv] == nil then
							total_probs[vv]=0
						end
						total_probs[vv] = total_probs[vv]+1
					end
				end
			end


			superposiciones[2] = total_probs
		else
			superposiciones[2] = 0
		end

		if left ==0  then
			local total_probs = {}
			local final = {}
			local x_q = x
			local y_q = y+1
			local sum = 0
			local chose_tittle = read_array(x_q,y_q,arr,false,nil)
			--*	local pprobs = {}
			local probs =  {read_array(x_q,y_q+1,arr,false,nil),
				read_array(x_q,y_q-1,arr,false,nil),
				read_array(x_q-1,y_q,arr,false,nil),
				read_array(x_q+1,y_q,arr,false,nil)}
			for k,v in pairs(probs) do


				if v~= 0  and v ~= nil then
					for kk,vv in pairs(configs[v]["vecinos"]) do

						sum = sum+1
						if total_probs[vv] == nil then
							total_probs[vv]=0
						end
						total_probs[vv] = total_probs[vv]+1
					end
				end
			end


			superposiciones[3] = total_probs
		else
			superposiciones[3] = 0
		end

		if right ==0  then
			local total_probs = {}
			local final = {}
			local x_q = x
			local y_q = y+1
			local sum = 0
			local chose_tittle = read_array(x_q,y_q,arr,false,nil)
			--*local pprobs = {}
			local probs =  {read_array(x_q,y_q+1,arr,false,nil),
				read_array(x_q,y_q-1,arr,false,nil),
				read_array(x_q-1,y_q,arr,false,nil),
				read_array(x_q+1,y_q,arr,false,nil)}
			for k,v in pairs(probs) do


				if v~= 0  and v ~= nil then
					for kk,vv in pairs(configs[v]["vecinos"]) do

						sum = sum+1
						if total_probs[vv] == nil then
							total_probs[vv]=0
						end
						total_probs[vv] = total_probs[vv]+1
					end
				end
			end


			superposiciones[4] = total_probs
		else
			superposiciones[4] = 0
		end

		-- colapzamos una celda vacia si es que la hay
		local random_chose = math.random(1,#superposiciones)

		for i=0,10  do
			if superposiciones[random_chose]==0 then
				random_chose = math.random(1,#superposiciones)
			end
		end
		--print(superposiciones[random_chose],"____ZZZZ")
		--print(superposiciones[random_chose])
		if superposiciones[random_chose]~=0 then
			local final=0
			local final_coords = {}
			local total_probs = {}
			local pprobs = {}
			if random_chose == 1 and superposiciones[random_chose]~= nil then
				local x_q = x
				local y_q = y+1
				local sum = 0
				local chose_tittle = read_array(x_q,y_q,arr,false,nil)
				--local pprobs = {}
				local probs =  {read_array(x_q,y_q+1,arr,false,nil),
					read_array(x_q,y_q-1,arr,false,nil),
					read_array(x_q-1,y_q,arr,false,nil),
					read_array(x_q+1,y_q,arr,false,nil)}

				for k,v in pairs(probs) do
					if v~= 0  and v ~= nil then
						for kk,vv in pairs(configs[v]["vecinos"]) do
							sum = sum+1
							if total_probs[vv] == nil then
								total_probs[vv]=0
							end
							total_probs[vv] = total_probs[vv]+1
						end
					end
				end
				for k,v in pairs(superposiciones[random_chose]) do
					--	if total_probs[k]== nil then
					if total_probs[k]== nil then
						total_probs[k] = 0
					end
					total_probs[k] = total_probs[k] + v
					sum = sum + v

					--	end
				end


				for k, v in pairs(total_probs) do
					pprobs[k] = (v*100)/sum
				end
				local finalcc = math.random(1,10000)/100

				local sum_probs = 0
				local prev = 0
				local max = 0
				local c
				for k,v in pairs(pprobs) do
					if v > max then
						max = v
						c = k
					end
				end
				if math.random(1,100) <= entropia then
					final=c
				else
					for k,v in pairs(pprobs) do
						sum_probs = sum_probs +v
						if finalcc <= sum_probs  and finalcc>= prev then
							final=k

						end
						prev = sum_probs-v
					end
				end

				final_coords={0,1}
			end
			if random_chose == 2 and superposiciones[random_chose]~= nil then
				local x_q = x
				local y_q = y-1
				local sum = 0
				local chose_tittle = read_array(x_q,y_q,arr,false,nil)
				--	local pprobs = {}
				local probs =  {read_array(x_q,y_q+1,arr,false,nil),
					read_array(x_q,y_q-1,arr,false,nil),
					read_array(x_q-1,y_q,arr,false,nil),
					read_array(x_q+1,y_q,arr,false,nil)}
				for k,v in pairs(probs) do
					if v~= 0  and v ~= nil then
						for kk,vv in pairs(configs[v]["vecinos"]) do
							sum = sum+1
							if total_probs[vv] == nil then
								total_probs[vv]=0
							end
							total_probs[vv] = total_probs[vv]+1
						end
					end
				end
				for k,v in pairs(superposiciones[random_chose]) do
					--	if total_probs[k]== nil then
					if total_probs[k]== nil then
						total_probs[k] = 0
					end
					total_probs[k] = total_probs[k] + v
					sum = sum + v

					--	end
				end
				--print(total_probs,"DDDD")
				for k, v in pairs(total_probs) do
					pprobs[k] = (v*100)/sum
				end
				local finalcc =  math.random(1,10000)/100
				local sum_probs = 0
				local prev = 0
				local c
				local max = 0
				for k,v in pairs(pprobs) do
					if v > max then
						max = v
						c = k
					end
				end
				if math.random(1,100) <= entropia then
					final=c
				else
					for k,v in pairs(pprobs) do
						sum_probs = sum_probs +v
						if finalcc <= sum_probs  and finalcc>= prev then
							final=k

						end
						prev = sum_probs-v
					end
				end

				final_coords={0,-1}
			end
			if random_chose == 3 and superposiciones[random_chose]~= nil then
				local x_q = x -1
				local y_q = y
				local sum = 0
				local chose_tittle = read_array(x_q,y_q,arr,false,nil)
				--	local pprobs = {}
				local probs =  {read_array(x_q,y_q+1,arr,false,nil),
					read_array(x_q,y_q-1,arr,false,nil),
					read_array(x_q-1,y_q,arr,false,nil),
					read_array(x_q+1,y_q,arr,false,nil)}
				for k,v in pairs(probs) do
					if v~= 0  and v ~= nil then
						for kk,vv in pairs(configs[v]["vecinos"]) do
							sum = sum+1
							if total_probs[vv] == nil then
								total_probs[vv]=0
							end
							total_probs[vv] = total_probs[vv]+1
						end
					end
				end
				for k,v in pairs(superposiciones[random_chose]) do
					--	if total_probs[k]== nil then
					if total_probs[k]== nil then
						total_probs[k] = 0
					end
					total_probs[k] = total_probs[k] + v
					sum = sum + v

					--	end
				end

				for k, v in pairs(total_probs) do
					pprobs[k] = (v*100)/sum
				end
				local finalcc =  math.random(1,10000)/100
				local sum_probs = 0
				local prev = 0
				local c
				local max = 0
				for k,v in pairs(pprobs) do
					if v > max then
						max = v
						c = k
					end
				end
				if math.random(1,100) <= entropia then
					final=c
				else
					for k,v in pairs(pprobs) do
						sum_probs = sum_probs +v
						if finalcc <= sum_probs  and finalcc>= prev then
							final=k

						end
						prev = sum_probs-v
					end
				end

				final_coords={-1,0}
			end
			if random_chose == 4 and superposiciones[random_chose]~= nil then
				local x_q = x +1
				local y_q = y
				local sum = 0
				local chose_tittle = read_array(x_q,y_q,arr,false,nil)
				--	local pprobs = {}
				local probs =  {read_array(x_q,y_q+1,arr,false,nil),
					read_array(x_q,y_q-1,arr,false,nil),
					read_array(x_q-1,y_q,arr,false,nil),
					read_array(x_q+1,y_q,arr,false,nil)}
				for k,v in pairs(probs) do
					if v~= 0  and v ~= nil then
						for kk,vv in pairs(configs[v]["vecinos"]) do
							sum = sum+1
							if total_probs[vv] == nil then
								total_probs[vv]=0
							end
							total_probs[vv] = total_probs[vv]+1
						end
					end
				end

				for k,v in pairs(superposiciones[random_chose]) do
					--	if total_probs[k]== nil then
					if total_probs[k]== nil then
						total_probs[k] = 0
					end
					total_probs[k] = total_probs[k] + v
					sum = sum + v

					--	end
				end
				for k, v in pairs(total_probs) do
					pprobs[k] = (v*100)/sum

				end
				--print(pprobs)
				local finalcc =  math.random(1,10000)/100
				local sum_probs = 0
				local prev = 0
				local c
				local max = 0
				for k,v in pairs(pprobs) do
					if v > max then
						max = v
						c = k
					end
				end
				if math.random(1,100) <= entropia then
					final=c
				else
					for k,v in pairs(pprobs) do
						sum_probs = sum_probs +v
						if finalcc <= sum_probs  and finalcc>= prev then
							final=k

						end
						prev = sum_probs-v
					end
				end
				final_coords={1,0}
			end
			--print(superposiciones[random_chose][1])
			--print(total_probs[1])
			if final_coords[1]~= nil and final ~= 0 then
				--	print(pprobs,final)
				--	print(final,final_coords)
				for k,v in pairs(indefinidos) do
					if (v[1] == current_x + final_coords[1]) and (v[2] == current_y +final_coords[2]) then
						table.remove(indefinidos,k)
						--print("used",v)
					end
				end
				parse(configs[final],read_array(current_x+final_coords[1],current_y+final_coords[2],ladrillos,false,nil))

				read_array(current_x+final_coords[1],current_y+final_coords[2],array,true,final) -- escribimos el estado actual de la celda
				current_x=current_x+final_coords[1]
				current_y=current_y+final_coords[2]
			else
			--		print("stuck")
				find=true
			end
		else
		--	print("no more posible k")
			find=true
		end
	else
		--	print("no more k")
		find = true
	end

	if find==true and #indefinidos>1 then
			--print(#indefinidos)

		local chose = math.random(1,#indefinidos)
		--index_to_coords()

		current_x = indefinidos[chose][1]
		current_y= indefinidos[chose][2]
		table.remove(indefinidos,chose)

		--print(current_x,current_y,current_tittle,chose)
		--current_tittle_p:Destroy()
	end


end
function wavecollapse.run()
	local waits = 0
	while #indefinidos>1 do
		wavecollapse.check_collapse(current_x,current_y,array)
		--print("D")
		waits= waits+1


	end
	print("end")
	local wd = 0
	local out = ""
	for k,v in pairs(array) do

		 out = out .. " ".. v
		 if y_s == wd then
			wd = 0
			print(out)
			out = ""
		  end
			wd = wd+1

	end
	return array, y_s,x_s
end
return wavecollapse
