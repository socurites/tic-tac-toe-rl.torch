--[[
-- Relay Memory for Tic Tac Toe
-- @maxMemory number the max number of experience to be remembered
-- @discount number the discount factor
]]--
function TicTacToeMemory(maxMemory, discount)
    local memory = {}
    
    -- Appends the experience to the memory.
    -- @param memoryInput table the newly expierenced state transition of an episode
    function memory.remember(memoryInput)      
      --[[
      for i=1, #memory do
        local target = memory[i]
        local other = memoryInput
        
        if ( target.action == other.action and
             target.reward == other.reward and
             target.gameOver == other.gameOver and
             torch.all(torch.eq(target.inputState, other.inputState)) and
             torch.all(torch.eq(target.nextState, other.nextState)) ) then        
          return nil
        end
      end
      ]]--
      
      table.insert(memory, memoryInput)
      if (#memory > maxMemory) then
          table.remove(memory, 1)
      end
    end
    
    -- Get mini-batch
    -- @param model
    -- @param batchSize
    -- @param numActions
    -- @param numStates
    function memory.getBatch(model, batchSize, numActions, numStates)
        local memoryLength = #memory
        local chosenBatchSize = math.min(batchSize, memoryLength)
        
        local inputs = torch.Tensor(chosenBatchSize, numStates):zero()
        local targets = torch.Tensor(chosenBatchSize, numActions):zero()
        
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
