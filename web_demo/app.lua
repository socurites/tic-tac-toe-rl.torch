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
            
    action = agentO.chooseAction(currState, epsilon)
    print('[agentO] : ' .. action)      
    
   res.send(tostring(action))
end)

app.listen()
