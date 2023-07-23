import random
import time
import math
from copy import deepcopy
import sys

NUM_ROWS = 6
NUM_COLS = 7

class State: 
    
    def __init__(self):
        # inicializa o estado do jogo
        self.board = [[0]*NUM_COLS for i in range(NUM_ROWS)] # [[0,0,0,...], [0,0,0,...], ...] estado inicial do tabuleiro (sem pecas)
        self.column_heights = [NUM_ROWS-1] * NUM_COLS # [5, 5, 5, 5, 5, 5, 5] index da proxima linha vazia de cada coluna
        self.available_moves = list(range(7)) # [0, 1, ..., 6] lista das colunas que ainda podem ser jogadas
        self.player = 1
        self.winner = -1 # -1 -> sem vencedor (durante o jogo), 0 -> empate, 1 -> jogador 1 ganhou, 2 -> jogador 2 ganhou
        
    #========================================================================================================================================#
    
    def check_line(self, row, col, dx, dy):
        # verifica se ha 4 pecas do mesmo jogador em linha
        player = self.board[row][col]
        if player == 0:
            return False

        for i in range(1, 4):
            if (row + i * dy >= NUM_ROWS or row + i * dy < 0 or
                col + i * dx >= NUM_COLS or col + i * dx < 0 or
                self.board[row + i * dy][col + i * dx] != player):
                return False
        return True
    
    #========================================================================================================================================#
        
    def move(self, column): 
        # funcao que executa um movimento dado o numero da coluna e retorna o novo estado
        # atualiza a lista de movimentos disponiveis, alturas das colunas, a vez do jogador e verifica se ha vencedores

        state_copy = deepcopy(self)
        
        height = state_copy.column_heights[column]
        state_copy.column_heights[column] = height
        state_copy.board[height][column] = self.player
        
        if height == 0: # se a altura da coluna for 0 (so tem uma posicao disponivel nessa coluna), remove a coluna da lista de movimentos disponiveis
            state_copy.available_moves.remove(column)
        else: # atualiza a altura da coluna
            state_copy.column_heights[column] = height - 1
        
        # verifica se ha vencedor e altera a vez do jogador
        state_copy.update_winner() 
        state_copy.player = 3 - self.player # atualiza a vez do jogador (1 -> (3-1) = 2, 2 -> (3-2) = 1)
        
        return state_copy
        

    #========================================================================================================================================#
 
    def update_winner(self):
        # funcao que verifica se ha vencedor e atualiza o atributo winner
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                if (self.check_line(row, col, 1, 0) or
                    self.check_line(row, col, 0, 1) or
                    self.check_line(row, col, 1, 1) or
                    self.check_line(row, col, 1, -1)):
                    self.winner = self.board[row][col]
                    return

        if len(self.available_moves) == 0:
            self.winner = 0

#########################################################################################################################################  

class ConnectFourGame:
    
    def __init__(self, player_1_, player_2_):
        self.state = State() # estado inicial
        self.player_1_ = player_1_ # jogador 1
        self.player_2_ = player_2_ # jogador 2
        
    def start(self):
        self.state = State()
        self.print_state()
        while True:

            if self.state.player == 1:
                print("  It is now 'X' turn.")
                self.player_1_(self)
            else:
                print("   It is now 'O' turn.")
                self.player_2_(self)
            
            self.print_state()
            

            if self.state.winner != -1:
                break
       
        if self.state.winner == 0:
            print("\t End of game! Draw!\n")
        else:
            print(f"\t End of game! Player {self.state.winner} wins!\n")
        
    
#########################################################################################################################################

    # funcao que imprime o estado do jogo
    def print_state(self):   
        for i in range(6):
            print("\t", end="")
            for j in range(7):
                piece = ""
                if (self.state.board[i][j]) == 1:
                    piece = "X"
                elif (self.state.board[i][j]) == 2:
                    piece = "O"
                else:
                    piece = "-"
                print("| " + str(piece), end=" ")
            print("|")
            
        print("\t+---+---+---+---+---+---+---+")
        print("\t  1   2   3   4   5   6   7 ")
        print()
        
#########################################################################################################################################   

def player_move(game):
    # funcao que executa o movimento do jogador humano
    # se o input for invalido, pede novamente um input valido
    
    move = (input("  Make a move by choosing your coordinates to play (1 to 7) : "))
    while (not move.isdigit()) or (int(move) < -1 or int(move) > 7) or ( int(move) - 1 not in game.state.available_moves):
        move = (input("  Make a move by choosing your coordinates to play (1 to 7) : "))
    print("\n")
    game.state = game.state.move(int(move) -1)
    
    
#########################################################################################################################################


############################
#                          #
# IMPLEMENTACAO DO MINIMAX # 
#                          #
############################


def execute_minimax_move(minmaxfunc, depth):
    def execute_minimax_move_aux(game):
        # funcao que executa o movimento do jogador minimax (minimax e minimax com cortes alpha-beta)
        print("   Thinking...\n")
        start_time = time.time()
        best_move = []
        best_eval = float('-inf')
        if minmaxfunc == minimax_alpha_beta:
            label = "Alpha-Beta"
        else:
            label = "Minimax"
        
        for move in game.state.available_moves: # executa os movimentos possiveis e calcula a avaliacao de cada um, escolhendo um dos com a melhor avaliacao
            move_choice = move
            new_state = game.state.move(move)
            
            if minmaxfunc == minimax_alpha_beta:
                new_state_eval,node_count = minmaxfunc(new_state, depth - 1, float('-inf'), float('+inf'), False, game.state.player,[0,0])
            else:
                new_state_eval,node_count = minmaxfunc(new_state, depth - 1, False, game.state.player,[0,0])
                
            if new_state_eval > best_eval:
                best_move = [(new_state,move_choice)]
                best_eval = new_state_eval
            elif new_state_eval == best_eval:
                best_move.append((new_state,move_choice))
        
        
        move_chosen = random.choice(best_move)
        game.state = move_chosen[0]
        best_choice = move_chosen[1]
        
        end_time = time.time()
        
        print(f"   {label} Move: Column {best_choice + 1} selected in {(end_time - start_time):.4}s")
        print(f"\t     Nodes Explored: {node_count[0]}")
        if minmaxfunc == minimax_alpha_beta:
            print(f"\t     Nodes Pruned: {node_count[1]}\n")
        
        
    return execute_minimax_move_aux

#========================================================================================================================================#

#minimax sem cortes alpha-beta
def minimax(state, depth, maximizing, player,node_count):
    
    node_count[0] += 1
    
    if depth == 0 or state.winner != -1:
        return evaluate_func(state) * (1 if player == 1 else -1) , node_count
    
    if maximizing:
        max_eval = float('-inf')
        for move in state.available_moves:
            new_state = state.move(move)
            eval,node_count = minimax(new_state, depth - 1, False, player,node_count)
            max_eval = max(max_eval, eval)
        return max_eval,node_count
    else:
        min_eval = float('inf')
        for move in state.available_moves:
            new_state = state.move(move)
            eval,node_count = minimax(new_state, depth - 1,True, player,node_count)
            min_eval = min(min_eval, eval)
        return min_eval,node_count

#========================================================================================================================================#

#minimax com cortes alpha-beta
def minimax_alpha_beta(state, depth, alpha, beta, maximizing, player,node_count):
    
    node_count[0] += 1
    
    if depth == 0 or state.winner != -1:
        return evaluate_func(state) * (1 if player == 1 else -1) , node_count
    
    if maximizing:
        max_eval = float('-inf')
        for move in state.available_moves:
            new_state = state.move(move)
            eval,node_count = minimax_alpha_beta(new_state, depth - 1, alpha, beta, False, player,node_count)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                node_count[1] += 1
                break
        return max_eval,node_count
    else:
        min_eval = float('inf')
        for move in state.available_moves:
            new_state = state.move(move)
            eval,node_count = minimax_alpha_beta(new_state, depth - 1, alpha, beta, True, player,node_count)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                node_count[1] += 1
                break
        return min_eval,node_count
    
#========================================================================================================================================#

# funcao que avalia as sequencias de 4 casas
def evaluate_seq(values):
    
    sum = 0
    if values.count(1) == 0 and values.count(2) == 3:
        sum -= 50
    if values.count(1) == 0 and values.count(2) == 2:
        sum -= 10
    if values.count(1) == 0 and values.count(2) == 1:
        sum -= 1
        
    if values.count(1) == 1 and values.count(2) == 0:
        sum += 1
    if values.count(1) == 2 and values.count(2) == 0:
        sum += 10
    if values.count(1) == 3 and values.count(2) == 0:
        sum += 50
    
    return sum

#========================================================================================================================================#

# funcao que avalia o estado do jogo
def evaluate_func(state):
    
    points = 0
    
    if state.winner == 1:
        return 512
    elif state.winner == 2:
        return -512
    elif state.winner == 0:
        return 0
    
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            # verifica a linha vertical
            if col < NUM_COLS - 3:
                points += evaluate_seq([state.board[row][col], state.board[row][col+1], state.board[row][col+2], state.board[row][col+3]])
            # verifica a linha horizontal
            if row < NUM_ROWS - 3 :
                points += evaluate_seq([state.board[row][col], state.board[row+1][col], state.board[row+2][col], state.board[row+3][col]])
            # verifica a diagonal superior
            if row < NUM_ROWS - 3 and col < NUM_COLS - 3: 
                points += evaluate_seq([state.board[row][col], state.board[row+1][col+1], state.board[row+2][col+2], state.board[row+3][col+3]])
            # verifica a diagonal inferior
            if row < NUM_ROWS - 3 and col > 3:
                points += evaluate_seq([state.board[row][col], state.board[row+1][col-1], state.board[row+2][col-2], state.board[row+3][col-3]])
 
    return points

######################################################################################################################################### 

#########################
#                       #
# IMPLEMENTACAO DO MTCS #
#                       #
#########################

#executa o movimento do mcts
def execute_mcts_move(lim):
    def execute_mcts_move_aux(game):
        print("   Thinking...\n")
        start_time = time.time()
        move_choice = MCTSPlayer(lim).select_move(game.state,game.state.player)
        move= move_choice[0]
        n_rollouts = move_choice[1]       
        game.state = game.state.move(move)
        end_time = time.time()
        print(f"   MCTS Move: Column {move + 1} selected in {(end_time - start_time):.4}s")
        print(f'\t     Number of Rollouts: {n_rollouts}\n')
        
    return execute_mcts_move_aux

#========================================================================================================================================#

# no do MCTS
class MCTSNode:
    global LIM
    # state -> estado do jogo
    # parent -> no pai
    # children -> dicionario com os filhos: chave -> movimento, valor -> no filho
    # wins -> numero de vitorias
    # visits -> numero de visitas
    # untried_moves -> lista com os movimentos nao testados
    def __init__(self, state, parent=None): 
        self.state = state
        self.parent = parent
        self.children = {}
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.available_moves.copy()

    def UCT_select_child(self, c=3):
        # usa a formula UCT para selecionar o filho
        # c -> constante de exploracao
        # devolve o no filho com maior pontuacao UCT
        if LIM > 4:
            c = 4
        
        best_child = None
        best_score = -1
        for move, child_node in self.children.items():
            UCT_score = child_node.wins / child_node.visits + c * math.sqrt(math.log(self.visits) / child_node.visits)
            if UCT_score > best_score:
                best_child = child_node
                best_score = UCT_score
        return best_child

    def expand(self):
        # Escolhe um movimento que ainda nao foi tentado e cria o filho
        move = random.choice(self.untried_moves)
        new_state = self.state.move(move)
        new_node = MCTSNode(new_state, self)
        self.children[move] = new_node
        self.untried_moves.remove(move)
        return new_node

    def back_prop(self, result):
        # faz backpropagation do resultado
        self.wins += result
        self.visits += 1
        if self.parent is not None:
            self.parent.back_prop(result)


    def is_fully_expanded(self):
        # devolve se todos os movimentos foram testados para este no
        return len(self.untried_moves) == 0

#========================================================================================================================================#

class MCTSPlayer:
    
    def __init__(self, time_limit=1.0):
        self.time_limit = time_limit

    def select_move(self, state,player):
        # seleciona o movimento a partir do estado atual
        root = MCTSNode(state)
        num_rollouts = 0
        start_time = time.time()
        while time.time() - start_time < self.time_limit:
            node = root
            # Select
            while node.is_fully_expanded() and len(node.children) > 0: # enquanto o no nao for folha
                node = node.UCT_select_child()
            # Expand
            if not node.is_fully_expanded():
                node = node.expand()
            # Rollout
            while node.state.winner == -1:
                node = node.expand()
                num_rollouts +=1
              
            if node.state.winner == 0 or node.state.winner != player:
                result = 0
            else:
                result = 1
            # Backpropagate
            node.back_prop(result)
            
            
        # escolhe o melhor no ,isto e, o no com mais visitas
        best_move = None
        best_visits = -1
        for move, child_node in root.children.items():
            if child_node.visits > best_visits:
                best_visits = child_node.visits
                best_move = move
            
        return best_move,num_rollouts


#########################################################################################################################################
LIM = 0

if __name__ == "__main__":
    
    ALG = sys.argv[1] # algoritmo a usar
    LIM = int(sys.argv[2]) # profundidade a usar (minimax e alpha-beta) ou limite de tempo (MCTS)

    if ALG == "MiniMax":
        game = ConnectFourGame(player_move, execute_minimax_move(minimax, LIM))
        game.start()

    if ALG == "Alpha-Beta":
        game = ConnectFourGame(player_move, execute_minimax_move(minimax_alpha_beta, LIM))
        game.start()
    
    if ALG == "MCTS":
        game = ConnectFourGame(player_move, execute_mcts_move(LIM))
        game.start()