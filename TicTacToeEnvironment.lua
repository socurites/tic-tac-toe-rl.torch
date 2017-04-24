function TicTacToeEnvironment(gridSize)
    local env = {}
    local state
    local currState
    local nextState
    local util = TicTacToeUtil()
    
    -- Resets the environment. Randomly initialise the fruit position (always at the top to begin with) and bucket.
    function env.reset()
      state = torch.Tensor(gridSize, gridSize):zero()
    end
        
    -- Returns the state of the environment.
    function env.observe()        
      return state
    end

    function env.drawState()      
    end    

    function env.getState()        
    end

    -- Returns the award that the agent has gained for being in the current environment state.
    function env.getReward()        
    end

    function env.isGameOver()
        
    end

    function env.updateState(action)
       
    end

    function env.act(action, stone)  
      local coord = util.coord(action)
      state[coord[1]][coord[2]] = stone
      
      local reward = 0
      local gameOver = false
      winningCond = util.checkWinState(state, stone)
      if ( winningCond == true ) then
        reward = 1
        gameOver = true
      elseif ( util.isAllMarked(state) ) then
        reward = 0.5
        gameOver = true        
      end
      
      return env.observe(), reward, gameOver
    end

    return env
end
