require 'nn'
require 'optim'
require 'TicTacToeMemory'
require 'TicTacToeEnvironment'
require 'TicTacToeRandomAgent'
require 'TicTacToeQLearningAgent'
require 'TicTacToeUtil'

local cmd = torch.CmdLine()
cmd:text('Training options')
cmd:option('-epsilon', 1, 'The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)')
cmd:option('-epsilonMinimumValue', 0.001, 'The minimum value we want epsilon to reach in training. (0 to 1)')
cmd:option('-numActions', 9, 'The number of actions.')
cmd:option('-epoch', 1, 'The number of games we want the system to run for.')
cmd:option('-hiddenSize', 50, 'Number of neurons in the hidden layers.')
cmd:option('-maxMemory', 10000, 'How large should the memory be (where it stores its past experiences).')
cmd:option('-batchSize', 50, 'The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.')
cmd:option('-gridSize', 3, 'The size of the grid that the agent is going to play the game on.')
cmd:option('-discount', 0.9, 'the discount is used to force the network to choose states that lead to the reward quicker (0 to 1)')
cmd:option('-savePrefix', 'tictactoc-', 'Save path for model')
cmd:option('-learningRate', 0.4)
cmd:option('-learningRateDecay', 1e-9)
cmd:option('-weightDecay', 0)
cmd:option('-momentum', 0.9)

local opt = cmd:parse(arg)

local epsilon = opt.epsilon
local epsilonMinimumValue = opt.epsilonMinimumValue
local numActions = opt.numActions
local epoch = opt.epoch
local hiddenSize = opt.hiddenSize
local maxMemory = opt.maxMemory
local batchSize = opt.batchSize
local gridSize = opt.gridSize
local numStates = gridSize * gridSize
local discount = opt.discount

-- Params for Stochastic Gradient Descent (our optimizer).
local sgdParams = {
    learningRate = opt.learningRate,
    learningRateDecay = opt.learningRateDecay,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    dampening = 0,
    nesterov = true
}

math.randomseed(os.time())


local util = TicTacToeUtil()

--[[ Runs one gradient update using SGD returning the loss.]] --
local function trainNetwork(model, inputs, targets, criterion, sgdParams)
    local loss = 0
    local x, gradParameters = model:getParameters()
    local function feval(x_new)
        gradParameters:zero()
        local predictions = model:forward(inputs)
        local loss = criterion:forward(predictions, targets)
        local gradOutput = criterion:backward(predictions, targets)
        model:backward(inputs, gradOutput)
        return loss, gradParameters
    end
    local _, fs = optim.sgd(feval, x, sgdParams)
    loss = loss + fs[1]
    return loss
end

-- Create the base model.
local modelO = nn.Sequential()
modelO:add(nn.Linear(numStates, hiddenSize))
modelO:add(nn.Tanh())
modelO:add(nn.Linear(hiddenSize, hiddenSize))
modelO:add(nn.Tanh())
modelO:add(nn.Linear(hiddenSize, numActions))
modelO:add(nn.LogSoftMax())

local modelX = nn.Sequential()
modelX:add(nn.Linear(numStates, hiddenSize))
modelX:add(nn.Tanh())
modelX:add(nn.Linear(hiddenSize, hiddenSize))
modelX:add(nn.Tanh())
modelX:add(nn.Linear(hiddenSize, numActions))
modelX:add(nn.LogSoftMax())

--[[
local modelO = torch.load("tictactoc-" .. "6000" .. ".t7")
]]--

-- Mean Squared Error for our loss function.
local criterion = nn.ClassNLLCriterion()



memoryO = TicTacToeMemory(maxMemory, discount)
memoryX = TicTacToeMemory(maxMemory, discount)
local env = TicTacToeEnvironment(gridSize)

local agentO = TicTacToeQLearningAgent(numActions, modelO, -1)
local agentX = TicTacToeQLearningAgent(numActions, modelX, 1)


local winOCount = 0
local winXCount = 0
local drawCount = 0
local gameResult = ''
for i = 1, epoch do
  -- Initialise the environment.
  env.reset()
  local errO = 0
  local errX = 0
  local gameOver = false
  local reward = 0
  
  -- The initial state of the environment.    
  local currState, nextState
  local action
  local experienceO = nil
  while (gameOver ~= true) do        
    currState = env.observe():clone()
    
    -- First        
    action = agentO.chooseAction(currState, epsilon)
    print('[agentO] : ' .. action)
    
    -- Update Enviroiment
    nextState, reward, gameOver = env.act(action, agentO.stone())
    nextState = nextState:clone()
    if ( nextState ~= currState ) then
      experienceO = {
          inputState = currState:view(-1),
          action = action,
          reward = reward,
          nextState = nextState:view(-1),
          gameOver = gameOver
      }
    end       
    
    -- Game over by player O
    if ( gameOver == true and experienceO ~= nil) then
      memoryO.remember(experienceO)                  
      if ( experienceO.reward == 1 ) then
        winOCount = winOCount + 1
        experienceX.reward = -1
        experienceX.gameOver = true
        gameResult = 'O Win'
        
        memoryX.remember(experienceX)
        experienceX = nil
      else
        drawCount = drawCount + 1
        gameResult = 'Draw'
        
        memoryX.remember(experienceX)
        experienceX = nil
      end 
      experienceO = nil
    else
      if ( experienceX ~= nil ) then
        memoryX.remember(experienceX)
      end
      -- Later 
      currState = env.observe():clone()
      
      action = agentX.chooseAction(currState, epsilon)
      print('[agentX] : ' .. action)
    
      -- Update Enviroiment
      nextState, reward, gameOver = env.act(action, agentX.stone())
      nextState = nextState:clone()
      if ( nextState ~= currState ) then
        experienceX = {
            inputState = currState:view(-1),
            action = action,
            reward = reward,
            nextState = nextState:view(-1),
            gameOver = gameOver
        }
      end
    
      -- Game over by player X
      if ( gameOver == true and experienceX ~= nil) then
        memoryX.remember(experienceX)        
        if ( experienceX.reward == 1 ) then
          winXCount = winXCount + 1
          experienceO.reward = -1
          experienceO.gameOver = true
          gameResult = 'X Win'
          
          memoryO.remember(experienceO)
          experienceO = nil
        else
          drawCount = drawCount + 1
          gameResult = 'Draw'
          
          memoryO.remember(experienceO)
          experienceO = nil
        end 
        experienceX = nil
      elseif ( gameOver ~= true and experienceX ~= nil ) then
        print("O-I")
        print(experienceO.inputState:view(3,3))
        print("O-N")
        print(experienceO.nextState:view(3,3))
        memoryO.remember(experienceO)
        experienceO = nil
      end
    end
    
    local inputs, targets = memoryO.getBatch(modelO, batchSize, numActions, numStates)    
    if ( inputs:size(1) == 1 ) then
        inputs_1 = inputs:view(-1)
    else
        inputs_1 = inputs
    end
    _, index = torch.max(targets, 2)
    targets_1 = index:view(-1)
    errO = errO + trainNetwork(modelO, inputs_1, targets_1, criterion, sgdParams)
    
    if ( #memoryX > 0 ) then
      local inputs, targets = memoryX.getBatch(modelX, batchSize, numActions, numStates)
      if ( inputs:size(1) == 1 ) then
          inputs_1 = inputs:view(-1)
      else
          inputs_1 = inputs
      end
      _, index = torch.max(targets, 2)
      targets_1 = index:view(-1)
      errX = errX + trainNetwork(modelX, inputs_1, targets_1, criterion, sgdParams)
    end


    -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
    if (epsilon > epsilonMinimumValue) then
        epsilon = epsilon * 0.999
    end
  end
  print(string.format("Epoch %d: [WinO Rate = %.2f WinX Rate = %.2f Draw Rate = %.2f] err = %.5f : err = %.5f : WinO count %d : WinX count %d : Draw Count %d", i, winOCount / i * 100, winXCount / i * 100, drawCount / i * 100, errO, errX, winOCount, winXCount, drawCount))
  print(nextState)
  
  if ( i > 0 and i % 3000 == 0 ) then
    torch.save(opt.savePrefix .. i .. '.t7', modelO)
  end
end