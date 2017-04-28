-- Load required packages
require 'nn'
require 'optim'


-- Initialise environment and state
env = {}

gridSize = 3

canvas = torch.Tensor(gridSize, gridSize):zero()


-- Create network
nbStates = gridSize * gridSize
hiddenSize = 20
nbActions = 9

model = nn.Sequential()
model:add(nn.Linear(nbStates, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, hiddenSize))
model:add(nn.ReLU())
model:add(nn.Linear(hiddenSize, nbActions))

-- epsilon-greedy exploration: select action
--[[ Helper function: Chooses a random value between the two boundaries.]] --
function randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
end


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

-- User First
currentState = canvas:clone()
while ( true ) do
    action = math.random(1, nbActions)
    coordX = coord(action)    
    if ( isMarkable(currentState, coordX) )  then
        break
    end
end



-- Update Enviroiment
currentState[coordX[1]][coordX[2]] = -1



-- check state by User
userWinCond = checkWinState(currentState, -1)

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

-- observe reward
reward = 0
winCount = 0
gameOver = false


if ( userWinCond == true ) then
    reward = -1
    gameOver = true
elseif ( isAllMarked(currentState) ) then
    reward = 0.5
    gameOver = true
else
    -- Bot Later
    epsilon = 1
    if (randf(0, 1) <= epsilon) then
        action = math.random(1, nbActions)
    else
        q = model:forward(currentState)
        _, index = torch.max(q, 1)
        action = index[1]
    end
    coordY = coord(action)
    if ( isMarkable(currentState, coordY) )  then
        -- Update Environment
        nextState = currentState:clone()
        nextState[coordY[1]][coordY[2]] = 1
        botWinCond = checkWinState(currentState, -1)
        if ( botWindCod == true ) then
            reward = 1
            winCount = winCount + 1
            gameOver = true
        elseif ( isAllMarked(currentState) ) then
            reward = 0.5
            gameOver = true
        end
    else
        reward = -10
        gameOver= true
    end
end


-- Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
epsilonMinimumValue = 0.001
if (epsilon > epsilonMinimumValue) then
    epsilon = epsilon * 0.999
end



-- Initialise replay memory
memory = {}


-- save an experience to replay memory
table.insert(memory, {
    inputState = currentState:view(-1),
    action = action,
    reward = reward,
    nextState = nextState:view(-1),
    gameOver = gameOver
});

-- choose mini-batch of transitions
batchSize = 10
memoryLength = #memory
chosenBatchSize = math.min(batchSize, memoryLength)
inputs = torch.Tensor(chosenBatchSize, nbStates):zero()
targets = torch.Tensor(chosenBatchSize, nbActions):zero()

i = 1
-- Choose a random memory experience to add to the batch.
randomIndex = math.random(1, memoryLength)
memoryInput = memory[randomIndex]

-- Calculate Q-value
if (memoryInput.gameOver) then
    target[memoryInput.action] = memoryInput.reward
else
   discount = 0.9  -- discount factor
   
   -- Gives us Q_sa for all actions
   target = model:forward(memoryInput.inputState):clone()
   
   -- reward + discount(gamma) * max_a' Q(s',a')
   -- We are setting the Q-value for the action to  r + γmax a’ Q(s’, a’). 
   -- The rest stay the same to give an error of 0 for those outputs.
   -- the max q for the next state.
   nextStateMaxQ = torch.max(model:forward(memoryInput.nextState), 1)[1]
   target[memoryInput.action] = memoryInput.reward + discount * nextStateMaxQ
end
-- Update the inputs and targets.
inputs[i] = memoryInput.inputState
targets[i] = target

-- Train the network which returns the error.
criterion = nn.MSECriterion()

sgdParams = {
    learningRate = 0.1,
    learningRateDecay = 1e-9,
    weightDecay = weightDecay,
    momentum = 0.9,
    dampening = 0,
    nesterov = true
}

loss = 0
x, gradParameters = model:getParameters()
function feval(x_new)
    gradParameters:zero()
    local predictions = model:forward(inputs)
    local loss = criterion:forward(predictions, targets)
    local gradOutput = criterion:backward(predictions, targets)
    model:backward(inputs, gradOutput)
    return loss, gradParameters
end

_, fs = optim.sgd(feval, x, sgdParams)
loss = loss + fs[1]
