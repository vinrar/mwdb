def dtw(vector1, vector2, cost1, cost2):
    assert len(vector1) == len(cost1)
    assert len(vector2) == len(cost2)
    row = len(vector1) + 1
    col = len(vector2) + 1
    cost1.reverse()
    dp = [[float('inf')] * col for i in range(row)]
    dp[row - 1][0] = 0
    for i in range(row - 2, -1, -1):
        for j in range(1, col):
            cost = 0
            if vector1[i] != vector2[j - 1]:
                cost = abs(cost1[i] - cost2[j - 1])
            dp[i][j] = cost + min(dp[i + 1][j], dp[i][j - 1], dp[i + 1][j - 1])

    return dp[0][-1]

# v2 = [(7, 0), (9, 0), (2, 0), (9, 0), (2, 0), (7, 0), (6, 0), (5, 0), (4, 0)]
# v1 = [(9, 0), (5, 0), (2, 0), (4, 0), (5, 0), (5, 0), (8, 0)]
# c2 = [7, 9, 2, 9, 2, 7, 6, 5, 4]
# c1 = [9, 5, 2, 4, 5, 5, 8]
# dtw(v1, v2, c1, c2)
