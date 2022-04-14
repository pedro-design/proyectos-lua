--encriptar
local key = "among"

local k = string.byte(key)

local msg = "test test testing"

print("🌽")



local function how(msg)
	local t = {}
	for i = 1, #msg do
		t[i] = string.byte(msg:sub(i, i) )
	end
return t
end

local function hmmm(k,msg)
	local t = {}
	local parrot = {}
	local no = 1
	for d,v in pairs(msg) do

		table.insert(t,v+ k[no])
		table.insert(parrot,string.char(v+k[no]))
		no = no+1
		if no == #k then
			no = 1

		end

	end
return t,parrot
end

local function what_say(k,msg)
	local t = {}
	local parrot = {}
	local no = 1
	for d,v in pairs(msg) do
		if v == tostring(v) then
			v = string.byte(v)
		end

		table.insert(t,v- k[no])
		table.insert(parrot,string.char(v-k[no]))
		no = no+1
		if no == #k then
			no = 1

		end

	end
return t,parrot
end

-- mayousculas de 65 a 90
--minusculas de 97 a 112
print(string.byte("a"),string.byte("z"),string.char("91"))

local t = how(msg)
local platano = how(key)
local out,its_working =hmmm(platano,t)

print(table.concat(its_working))
local rb,ms =what_say(platano,its_working)
print(table.concat(ms))
print(table.concat(out))
io.read()