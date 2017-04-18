--[[
            Torch translation of the keras example found here (written by Eder Santana).
            https://gist.github.com/EderSantana/c7222daa328f0e885093#file-qlearn-py-L164

            Example of Re-inforcement learning using the Q function described in this paper from deepmind.
            https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

            The agent plays a game of catch. Fruits drop from the sky and the agent can choose the actions
            left/stay/right to catch the fruit before it reaches the ground.
]] --

require 'nn'
require 'optim'

local cmd = torch.CmdLine()
cmd:text('Training options')
cmd:option('-epsilon', 1, 'The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)')
cmd:option('-epsilonMinimumValue', 0.001, 'The minimum value we want epsilon to reach in training. (0 to 1)')
cmd:option('-nbActions', 9, 'The number of actions. Since we only have left/stay/right that means 3 actions.')
cmd:option('-epoch', 20000, 'The number of games we want the system to run for.')
cmd:option('-hiddenSize', 50, 'Number of neurons in the hidden layers.')
cmd:option('-maxMemory', 10000, 'How large should the memory be (where it stores its past experiences).')
cmd:option('-batchSize', 50, 'The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.')
cmd:option('-gridSize', 3, 'The size of the grid that the agent is going to play the game on.')
cmd:option('-discount', 0.9, 'the discount is used to force the network to choose states that lead to the reward quicker (0 to 1)')
cmd:option('-savePath', 'TorchQLearningModel.t7', 'Save path for model')
cmd:option('-learningRate', 0.4)
cmd:option('-learningRateDecay', 1e-9)
cmd:option('-weightDecay', 0)
cmd:option('-momentum', 0.9)

local opt = cmd:parse(arg)

local epsilon = opt.epsilon
local epsilonMinimumValue = opt.epsilonMinimumValue
local nbActions = opt.nbActions
local epoch = opt.epoch
local hiddenSize = opt.hiddenSize
local maxMemory = opt.maxMemory
local batchSize = opt.batchSize
local gridSize = opt.gridSize
local nbStates = gridSize * gridSize
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

--[[ Helper function: Chooses a random value between the two boundaries.]] --
local function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end

--[[ The memory: Handles the internal memory that we add experiences that occur based on agent's actions,
--   and creates batches of experiences based on the mini-batch size for training.]] --
local function Memory(maxMemory, discount)
    local memory = {}
    -- Appends the experience to the memory.
    function memory.remember(memoryInput)
        table.insert(memory, memoryInput)
        if (#memory > maxMemory) then
            -- Remove the earliest memory to allocate new experience to memory.
            table.remove(memory, 1)
        end
    end
    function memory.getBatch(model, batchSize, nbActions, nbStates)
        -- We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        -- batch we can (at the beginning of training we will not have enough experience to fill a batch).
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)
        local inputs = torch.Tensor(chosenBatchSize, nbStates):zero()
        local targets = torch.Tensor(chosenBatchSize, nbActions):zero()
        --Fill the inputs and targets up.
        for i = 1, chosenBatchSize do
            -- Choose a random memory experience to add to the batch.
            local randomIndex = math.random(1, memoryLength)
            local memoryInput = memory[randomIndex]
            local target = model:forward(memoryInput.inputState):clone()
            --Gives us Q_sa, the max q for the next state.
            local nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
            if (memoryInput.gameOver) then
                target[memoryInput.action] = memoryInput.reward
            else
                -- reward + discount(gamma) * max_a' Q(s',a')
                -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). The rest stay the same
                -- to give an error of 0 for those outputs.
                target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
            end
            -- Update the inputs and targets.
            inputs[i] = memoryInput.inputState
            targets[i] = target
        end
        return inputs, targets
    end
    return memory
end

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
local model = nn.Sequential()
model:add(nn.Linear(nbStates, hiddenSize))
model:add(nn.Tanh())
model:add(nn.Linear(hiddenSize, hiddenSize))
model:add(nn.Tanh())
model:add(nn.Linear(hiddenSize, nbActions))
model:add(nn.LogSoftMax())




-- Mean Squared Error for our loss function.
local criterion = nn.ClassNLLCriterion()

local memory = Memory(maxMemory, discount)



-- return int to [x][y] coordinate
-- [1 2 3
--  4 5 6
--  7 8 9 ]
function coord(num)
    if ( num <= 3 ) then
        return {1, num}
    elseif ( num <= 6 ) then
        return {2, num-3}
    else
        return {3, num-6}
    end
end

-- Win Conditions
function checkWinState(canvas, stone)
    -- horizontal
    for i = 1, 3 do
        if ( canvas[i][1] == stone and canvas[i][2] == stone and canvas[i][3] == stone ) then
            return true
        end
    end
    -- vertical
    for i = 1, 3 do
        if ( canvas[1][i] == stone and canvas[2][i] == stone and canvas[3][i] == stone ) then
            return true
        end
    end
    -- diagonal
    if ( canvas[1][1] == stone and canvas[2][2] == stone and canvas[3][3] == stone ) then
        return true
    end
    if ( canvas[1][3] == stone and canvas[2][2] == stone and canvas[3][1] == stone ) then
        return true
    end
    return false
end


function isMarkable(canvas, coord)
    if ( currentState[coord[1]][coord[2]] == 0 ) then
        return true
    end
    return false
end

function isAllMarked(canvas)
    for i= 1, 3 do
        for j = 1, 3 do
            if ( canvas[i][j] == 0 ) then
                return false
            end
        end
    end
    return true     
end



local winCount = 0
local drawCount = 0
local gameResult = ''
for i = 1, epoch do
    -- Initialise the environment.
    local err = 0
    local gameOver = false
    local reward = 0
    -- The initial state of the environment.
    local canvas = torch.Tensor(gridSize, gridSize):zero()
    currentState = canvas:clone()
    nextState = canvas:clone()
    while (gameOver ~= true) do
        local reRandom = false
        currentState = nextState:clone()
        -- User First
        while ( true ) do
            action = math.random(1, nbActions)
            coordX = coord(action)    
            if ( isMarkable(currentState, coordX) )  then
                break
            end 
        end
        print('[user] random: ' .. action)
        -- Update Enviroiment
        currentState[coordX[1]][coordX[2]] = -1
        -- check state by User
        userWinCond = checkWinState(currentState, -1)
        if ( userWinCond == true ) then
            reward = -1
            gameOver = true
            gameResult = 'Lose'
        elseif ( isAllMarked(currentState) ) then
            reward = 0.5
            drawCount = drawCount + 1
            gameOver = true
            gameResult = 'Draw'
        else
            -- Bot Later
            if (randf(0, 1) <= epsilon) then
                while ( true ) do
                    action = math.random(1, nbActions)
                    coordY = coord(action)    
                    if ( isMarkable(currentState, coordY) )  then
                        break
                    end                     
                end
                print('[bot] random: ' .. action)
            else
                q = model:forward(currentState:view(-1))
                 _, index = torch.sort(q, 1)
                for j = 1, 9 do
                    action = index[-j]
                    coordY = coord(action)
                    if ( isMarkable(currentState, coordY) )  then
                        break
                    end            
                end
                print('[bot]qval: ' .. action)
            end
            -- Update Environment
            nextState = currentState:clone()
            nextState[coordY[1]][coordY[2]] = 1
            botWinCond = checkWinState(nextState, 1)
            if ( botWinCond == true ) then
                reward = 1
                winCount = winCount + 1
                gameOver = true
                gameResult = 'Win'
            elseif ( isAllMarked(nextState) ) then
                reward = 0.5
                drawCount = drawCount + 1
                gameOver = true
                gameResult = 'Draw'
            end
            --if ( reRandom == true ) then
            --    reward = reward - 1.0
            --end
--            else
  --              nextState = currentState:clone()
    --            nextState[coordY[1]][coordY[2]] = -3
      --          reward = -3
        --        gameOver= true
        --    end
        end
        -- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
        if (epsilon > epsilonMinimumValue) then
            epsilon = epsilon * 0.999
        end

        if ( nextState ~= currentState ) then
            memory.remember({
                inputState = currentState:view(-1),
                action = action,
                reward = reward,
                nextState = nextState:view(-1),
                gameOver = gameOver
            })
            -- Update the current state and if the game is over.        
            -- We get a batch of training data to train the model.
            local inputs, targets = memory.getBatch(model, batchSize, nbActions, nbStates)
            if ( inputs:size(1) == 1 ) then
                inputs_1 = inputs:view(-1)
            else
                inputs_1 = inputs
            end
            _, index = torch.max(targets, 2)
            targets_1 = index:view(-1)
            -- Train the network which returns the error.
            err = err + trainNetwork(model, inputs_1, targets_1, criterion, sgdParams)
        end
        currentState = nextState:clone()
    end
    print(string.format("Epoch %d : [%s] [Win Rate = %.2f Draw Rate = %.2f] err = %f : Win count %d : Draw Count %d", i, gameResult, winCount / i * 100, drawCount / i * 100, err, winCount, drawCount))
    print(currentState)
end
torch.save(opt.savePath, model)
print("Model saved to " .. opt.savePath)
