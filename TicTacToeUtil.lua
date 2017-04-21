function TicTacToeUtil()
  local util = {}
  
  --[[ Helper function: Chooses a random value between the two boundaries.]] --
  function util.randf(s, e)
    return (math.random(0, (e - s) * 9999) / 10000) + s;
  end
  
  -- Return int to [x][y] coordinate
  -- [1 2 3
  --  4 5 6
  --  7 8 9]
  -- @param num int
  -- @return [x][y] coordinate
  function util.coord(num)
    if ( num <= 3 ) then
        return {1, num}
    elseif ( num <= 6 ) then
        return {2, num-3}
    else
        return {3, num-6}
    end
  end
  
  -- Check if the action is actionable
  function util.isActionable(canvas, action)
    local coord = util.coord(action)
    
    if ( canvas[coord[1]][coord[2]] == 0 ) then
        return true
    end
    return false
  end
  
  -- Check winning status
  function util.checkWinState(canvas, stone)
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
  
  function util.isAllMarked(canvas)
    for i= 1, 3 do
      for j = 1, 3 do
        if ( canvas[i][j] == 0 ) then
          return false
        end
      end
    end
    return true     
  end

  return util
end