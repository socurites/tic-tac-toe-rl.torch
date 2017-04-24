require 'TicTacToeUtil'

function TicTacToeQLearningAgent(numActions, model, stone)
  local agent = {}
  local util = TicTacToeUtil()
  
  -- Choose actoin based on epsilon-greedy strategy
  -- @param state
  -- @param epsilon
  -- @return
  function agent.chooseAction(state, epsilon)
    local action
    if (util.randf(0, 1) <= epsilon) then
      while ( true ) do
        action = math.random(1, numActions)                    
        if ( util.isActionable(state, action) ) then
          print("random")
          break
        end                     
      end
     else
       q = model:forward(state:view(-1))
       _, index = torch.sort(q, 1)
       for j = 1, 9 do
        action = index[-j]                    
        if ( util.isActionable(state, action) ) then
          print("action: " .. action)
          print(state:view(3,3))
          break
        end            
       end      
    end
    return action
  end
  
  function agent.stone()
    return stone
  end
  
  return agent
end