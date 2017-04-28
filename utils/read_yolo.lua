local io = require("io")
local http = require("socket.http")
local ltn12 = require("ltn12")

local clock = os.clock
function sleep(n)  -- seconds
  local t0 = clock()
  while clock() - t0 <= n do end
end

local function split(s, delimiter)
    local result = {};
    for match in (s..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match);
    end
    return result;
end




while true do
  sleep(5)
  
  for file in io.popen("ls " .. yolo_dir):lines() do
    if string.find(file, "%.csv$") then       
    
      fname = split(file, "%.")[1]
      print (fname)
      if ( tonumber(fname) > lastIndex ) then
        lastIndex = tonumber(fname)
        
        local f = io.open(yolo_dir .. file, "r");
        local content = f:read("*all")
        f:close()
      
        print(content)
      end
    end
  end
  
end


