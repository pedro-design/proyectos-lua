-- esta es una libreria multitasking para lua 
local maid = {
	tasks = {},
	running_coroutines = {}
}

-- recursive copy a metatable
function maid:deepCopy(original)
	local copy = {}
	for k, v in pairs(original) do
		if type(v) == "table" then
			v = maid:deepCopy(v)
		end
		copy[k] = v
	end
	return copy
end

function maid:new(o, task)
	-- create a new maid multitask system
	o = o or {}
	setmetatable(o, self)
	o.tasks = {}
	o.running_coroutines = {}
	return o
end

function maid:add_task(func, interval, repeats)
	-- create a new task , args (function, time_interval_in seconds,repeats(optional))
	repeats = repeats or -1
	local t = { f = func, interval = interval, repeats = repeats }
	table.insert(self.tasks, t)
	return 1
end

function maid:clean()
	-- remove finished tasks
	for k, task in pairs(self.tasks) do
		if task.repeats == 0 then
			table.remove(self.tasks, k)
		end
	end
end

function maid:run()
	-- run the system
	local current_time = os.clock()
	for k, task in pairs(self.tasks) do
		if (not self.running_coroutines[k]) and (task.repeats > 0 or task.repeats == -1) then
			local co = coroutine.create(task.f)
			coroutine.resume(co)
			self.running_coroutines[k] = { co = co, interval = task.interval, last_run = current_time }
		end
	end
	for k, coro in pairs(self.running_coroutines) do
		if coroutine.status(coro.co) == "dead" and (coro.last_run + coro.interval) <current_time then
			self.running_coroutines[k] = nil
			if self.tasks[k] ~= nil  then
				if self.tasks[k].repeats~= -1 then
					self.tasks[k].repeats = self.tasks[k].repeats - 1
				end
			end
		elseif (coro.last_run + coro.interval) < current_time and coroutine.status(coro.co) == "dead" then
			coroutine.resume(coro.co)
			coro.last_run = current_time
			local task = self.deepCopy(self.tasks[k])
			table.insert(self.tasks, task)
		end
	end
	self:clean()
end
-- example code
--local m = require("path to the module")

--local function h()
--	print("hi from the task 1")
--end

--local function h2()
--	print("hi from the task 2")
--end

--local function h3()
--	print("hi from the task 3")
--end

--local maid = m:new(m:deepCopy(m)) --
--if the task is created, return 1 the function
--print(maid:add_task(h,1,10)) -- run 10 times each second
--print(maid:add_task(h2,3)) -- run infinite times each 3 seconds
--print(maid:add_task(h3,5)) -- ron infinite times each 5 seconds

--while #maid.tasks>0 do
--	maid:run()
--	wait()
--end
-- if all task ended, continue the code
--print("all task ended")


return maid
