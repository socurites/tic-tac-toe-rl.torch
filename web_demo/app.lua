package.path = package.path .. ";../?.lua"
require 'nn'
require 'TicTacToeEnvironment'
require 'image'
require 'TicTacToeUtil'
require 'TicTacToeQLearningAgent'



function split(s, delimiter)
    result = {};
    for match in (s..delimiter):gmatch("(.-)"..delimiter) do
        table.insert(result, match);
    end
    return result;
end

--[[
--|1|2|3|
--|6|5|4|
--|7|8|9|
]]--
local function toCanvas(state, outFile)
  outFile:write("|" .. state[1][1] .. "|" .. state[1][2] .. "|" .. state[1][3] .. "|\n")
  outFile:write("|" .. state[2][1] .. "|" .. state[2][2] .. "|" .. state[2][3] .. "|\n")
  outFile:write("|" .. state[3][1] .. "|" .. state[3][2] .. "|" .. state[3][3] .. "|\n")
end


bot_file = io.open("bot.out", "w")
user_file = io.open("user.out", "w")



local cmd = torch.CmdLine()
cmd:text('Training options')
cmd:option('-epoch', 42000, 'The epoch of pre-trained model')
cmd:option('-gridSize', 3, 'The size of the grid that the agent is going to play the game on.')
cmd:option('-numActions', 9, 'The number of actions.')

local opt = cmd:parse(arg)
local numActions = opt.numActions
local epoch = opt.epoch
local gridSize = opt.gridSize
local epsilon = 0.0

if ( epoch == 10 ) then epsilon = 1 end


math.randomseed(os.time())

local util = TicTacToeUtil()

local model = torch.load("../tictactoc-" .. epoch .. "-O.t7")

local env = TicTacToeEnvironment(gridSize)

local agentO = TicTacToeQLearningAgent(numActions, model, -1)





local app = require('waffle')


app.get('/', function(req, res)   
   res.render('./index.html')
end)


app.get('/get_move', function(req, res)
    local canvas_str = req.url.args.canvas
    local canvas = split(canvas_str, ",")
    local currState = torch.Tensor(3,3)
    currState[1][1] = canvas[1]
    currState[1][2] = canvas[2]
    currState[1][3] = canvas[3]
    currState[2][1] = canvas[4]
    currState[2][2] = canvas[5]
    currState[2][3] = canvas[6]
    currState[3][1] = canvas[7]
    currState[3][2] = canvas[8]
    currState[3][3] = canvas[9]
    
    bot_file:write("Current state: \n")
    toCanvas(currState, bot_file)
    
    bot_file:write("Selecting the best position ... \n")
    action = agentO.chooseAction(currState, epsilon)
    bot_file:write("Action " .. action .. " was selected \n\n")
    print('[agentO] : ' .. action)      
    
   res.send(tostring(action))
end)


local yolo_dir = '/tmp/ebs_tictactoe/'

local lastIndex = -1

app.get('/user_move', function(req, res)
    user_file:write("Waiting for user input ...\n")
    
    local content = ''
    for file in io.popen("ls " .. yolo_dir):lines() do
      if string.find(file, "%.csv$") then             
        fname = split(file, "%.")[1]
        
        if ( tonumber(fname) > lastIndex ) then
          lastIndex = tonumber(fname)
          
          local f = io.open(yolo_dir .. file, "r");
          content = f:read("*all")
          f:close()
          
          
          local canvas_str = content
          local canvas = split(canvas_str, ",")
          local currState = torch.Tensor(3,3)
          currState[1][1] = canvas[1]
          currState[1][2] = canvas[2]
          currState[1][3] = canvas[3]
          currState[2][1] = canvas[4]
          currState[2][2] = canvas[5]
          currState[2][3] = canvas[6]
          currState[3][1] = canvas[7]
          currState[3][2] = canvas[8]
          currState[3][3] = canvas[9]
          
          user_file:write("User decieded action\n")
          toCanvas(currState, user_file)
          user_file:write("\n")
        end
      end
    end
    
    res.send(content)
end)

app.listen()
