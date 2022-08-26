# -*- coding: utf-8/ -*-
"""
Created on Fri Jun 10 22:49:44 2022

@author: zachh
"""

import numpy as np
import random
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense,Activation
from tensorflow import keras
from collections import deque


#==================================================================================
#    THINGS TO INCLUDE
# 1) if player 1 wins with a flush, that needs to be printed
# 2) the players should be able to request to see their hand at various times
# 3) high cards other than the first high card are not being taken into account
# 4) need a computer vs computer gameplay function
# 5) 

#=====================================================================================
#hands go from 'high number' and 'spade,club,heart,diamond'
deck = np.arange(52) #the deck of cards
#deck[0] = 1
#
num_players = 2
stack_size = 10000 #amount each player starts off with
big_blind = 50
small_blind = 25


#=================================================================================================
#======================== NEURAL NETWORK =========================================================
#=================================================================================================
#assigning weights to computer 1 and computer 2
weights_c1 = []
weights_c2 = []
pre_act_c1 = []
pre_act_c2 = []
neurons_c1 = []
neurons_c2 = []

n_layers = 2
epsilon = 1
max_ep = 1
min_ep = .01
decay = 0.01
#input neurons for: hand value, number and suit of each card(10vals tot)[==0 if cards dont exist]
#                   position, preflop move, postflop move, postturn move, postriver move,
#                   preflop bet_amt, postflop bet_amt, PT bet_amt, PR bet_amt, stack size, opp's stack size,
#                   pot size, 
input_neurons = 47
# 8  output neurons for: fold,call,raise 1/4,raise1/2, raise 3/4, raise pot, raise 5/4, all in
output_neurons = 8
n_neurons = [47,8]

sizing = np.insert(n_neurons,0,input_neurons)
for i in range(n_layers): #initializing weights with given NN architecture
    weights_c1.append(np.random.uniform(-1,1,size=(sizing[i+1],sizing[i])))
    weights_c2.append(np.random.uniform(-1,1,size=(sizing[i+1],sizing[i])))
    pre_act_c1.append(np.zeros(n_neurons[i]))
    pre_act_c2.append(np.zeros(n_neurons[i]))
    neurons_c1.append(np.zeros(n_neurons[i]))
    neurons_c2.append(np.zeros(n_neurons[i]))
    
def comp_network(input_neurons_shape,output_neurons_shape): #creating tensorflow/keras network
    #both inputs shoudl be tuples with the shape of the number of input and output nerusons
    #which correspond to number of incoming data points and possible actions
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, input_shape=input_neurons_shape, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(16, activation='relu', kernel_initializer=init))
    model.add(tf.keras.layers.Dense(output_neurons_shape, activation='linear', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model
    
def comp_vs_comp_inti():
    global input_neurons
    global output_neurons
    input_shape = (input_neurons,1)
    output_shape = (output_neurons,1)
    comp1_network = comp_network(input_shape,output_shape)
    comp2_network = comp_network(input_shape,output_shape)
    comp1_target_network = comp_network(input_shape,output_shape)
    comp2_target_network = comp_network(input_shape,output_shape)
    comp1_target_network.set_weights(comp1_network.get_weights())
    comp2_target_network.set_weights(comp2_network.get_weights())
    
    
    
def sigmoid(x,deriv):
    if deriv == False: #returns sigmoid func
        return(1/(1+np.exp(x)))
    else:
        sig = 1/(1+np.exp(x)) #returns sigmoid derivative
        return(sig*(1-sig))

def feedforward(state,weights,pre_act,neurons):
    
    for layer in range(n_layers):
        if layer == 0:
            pre_act[0] = np.matmul(weights[0],state)
            neurons[0] = sigmoid(pre_act[0],deriv=False)
        else:
            pre_act[layer] = np.matmul(weights[layer],neurons[layer-1])
            neurons[layer] = sigmoid(pre_act[layer],deriv=False)
    return(neurons)
    

#=================================================================================================
#======================== functions to play the game =============================================
def DealCards(deck,num_players):
    
    HoleCards = np.zeros((num_players,2))
    
    for i in range(num_players):
        HoleCards[i][0] = random.choice(deck)
        deck = np.delete(deck,np.where(deck == HoleCards[i][0])[0][0])
        HoleCards[i][1] = random.choice(deck)
        deck = np.delete(deck,np.where(deck == HoleCards[i][1])[0][0])

    return(deck,HoleCards)
        
def Flop(deck): 
    flop = np.zeros((3))
    for i in range(3):
        flop[i] = random.choice(deck)
        deck = np.delete(deck, np.where(deck == flop[i])[0][0])
    return(deck,flop)

def TurnRiver(deck): #can be used for both the turn and river
    turn = random.choice(deck)
    deck = np.delete(deck, np.where(deck == turn)[0][0])
    return(deck,turn)


def Suit(cards):
    suit = []
    decki = np.arange(52)
    spades = np.split(decki,4)[0]
    clubs = np.split(decki,4)[1]
    hearts = np.split(decki,4)[2]
    diamonds = np.split(decki,4)[3]
    if isinstance(cards,int): #so i can find suit of an individual card
        lookS = np.where(cards==spades)
        lookC = np.where(cards==clubs)
        lookH = np.where(cards==hearts)
        lookD = np.where(cards==diamonds)
        look = [lookS,lookC,lookH,lookD]
        for i in range(len(look)):
            x = len(look[i][0])
            if x == 1: return(i+1) #returns the suit
    for j in range(len(cards)):
        for k in range(len(spades)):
            if cards[j] == spades[k]: #spades given value of 1
                suit = np.append(suit,1)
            elif cards[j] == clubs[k]: #clubs given value of 2
                suit = np.append(suit,2)
            elif cards[j] == hearts[k]: #hearts given value of 3
                suit = np.append(suit,3)
            elif cards[j] == diamonds[k]: #diamonds given value of 4
                suit = np.append(suit,4)
    return(suit)


def Cards(hole_cards,flop,turn,river): #set flop/turn/river == 99 if you haven reached it yet
    "returns all cards on board and hole cards combined"
    cards = hole_cards
    if isinstance(flop,int):
        cards = hole_cards
    elif len(flop) > 2 and turn == 99:
        cards = np.append(cards,flop)
    elif turn != 99 and river == 99:
        cards = np.append(cards,flop)
        cards = np.append(cards,turn)
    else:
        cards = np.append(cards,flop)
        cards = np.append(cards,turn)
        cards = np.append(cards,river)
    return(cards)

#one pair
#two pair
#three of a kind
#straigt
#flush
#full house
#four of a kind
def Evaluate(card_vals):
    unique = np.unique(card_vals,return_counts=True) #finding #occurences of each value
    cards_and_occurs = np.asarray(unique) #making into ndarray
    ones = np.where(cards_and_occurs[1] == 1)[0] #finding where a card only occurs once
    cards_and_occurs = np.delete(cards_and_occurs,ones,axis=1) #deleting card if only one exists
    cards_and_occurs = np.unique(cards_and_occurs,return_index=True,axis=1)[0] #returns only unique columns
    return(cards_and_occurs)
    
def Classify(cards,cards_and_occurs,suit_class):
    "classifies which hand is best"
    #value structure needs to be decided. current ones are placeholders
    cao = cards_and_occurs.copy()
    card_vals = CardVal(cards)
    if np.isin(4,cao[1]): #four of a kind -------------------------
        value = 210 #should be a higher val than all worse hands
        x = np.where(cao[1] == 4)[0][0]
        card = cao[0][x] #card number of four of a kind
        value += card
        
    elif np.isin(3,cao[1]) and np.isin(2,cao[1]): #full house -----------------------
    #needs to be improved so that if player has two pairs and 3 of a kind, it will classify with the larger pair
        value = 180 #
        x = np.where(cao[1] == 3)[0][0] #loc of 3 of a kind
        y = np.where(cao[1] == 2)[0][0] #loc of pair
        card3 = cao[0][x]
        card2 = cao[0][y]
        value += card3 + card2
        
    elif FlushClass(card_vals,suit_class)[0]: #flush -----------------------
        value = 150
        card = FlushClass(card_vals,suit_class)[1]
        value += card
        
    elif StraightClass(card_vals)[0]: #straight ------------------
        value = 120
        card = StraightClass(card_vals)[1]
        value += card
        
    elif np.isin(3,cao[1]): #three of a kind ------------------
        value = 90
        x = np.where(cao[1] == 3)[0][0]
        card = cao[0][x]
        value += card
        
    elif TwoClass(cao)[0]: #two pair ----------------------------------
        value = 60
        card1 = TwoClass(cao)[1]
        card2 = TwoClass(cao)[2]
        value += card1 + card2
        
    elif len(cao[0]) == 1:  #one pair -------------------------
        value = 30
        card = cao[0][0]
        value += card
        
    else: # high card ----------------------------------------------
        value = 0
        high_card = np.max(card_vals)
        value += high_card
        
    return(value)
        
        
def TwoClass(cards_and_occurs): #if two pair exists, if 3 pair, take highest 2
    cao = cards_and_occurs #shortening
    length = len(cao[0]) 
    if length == 0:
        return(False,0,0)
    if length == 1:
        return(False,0,0)
    if length == 2:
        return(True,cao[0][0],cao[0][1])
    if length == 3:
        card1 = cao[0][-1]
        card2 = cao[0][-2]
        return(True,card1,card2)
        
    
        
def StraightClass(cardval): #classifying if a straight exists, and returns true/false as well high card
    sort = np.sort(cardval) #puts card vals in ascending order
    length = len(cardval)
    if length < 5: return(False,0)
    sort = np.unique(sort) #removing duplicates
    if len(sort) == 7:
        test1 = sort[:5]
        test2 = sort[1:6]
        test3 = sort[2:]
        test1a,test2a,test3a = 0, 0, 0
        for i in range(len(test1)-1):
            if test1[i] == test1[i+1] - 1:
                test1a += 1
            if test2[i] == test2[i+1] - 1:
                test2a += 1
            if test3[i] == test3[i+1] - 1:
                test3a += 1
        ans = np.asarray([test1a,test2a,test3a])
        tests = np.asarray([test1,test2,test3])
        if np.max(ans) == 4:
            Straight = True
            str_cards = tests[np.argmax(ans)]
            high = np.max(str_cards)
            return(Straight,high)
        else:
            return(False,0)
    if len(sort) == 6:
        test1 = sort[:5]
        test2 = sort[1:]
        test1a,test2a = 0, 0
        for i in range(len(test1)-1):
            if test1[i] == test1[i+1] - 1:
                test1a += 1
            if test2[i] == test2[i+1] - 1:
                test2a += 1
        ans = np.asarray([test1a,test2a])
        tests = np.asarray([test1,test2])
        if np.max(ans) == 4:
            Straight = True
            str_cards = tests[np.argmax(ans)]
            high = np.max(str_cards)
            return(Straight,high)
        else:
            return(False,0)
    if len(sort) == 5:
        test1 = sort
        test1a = 0
        for i in range(len(test1)-1):
            if test1[i] == test1[i+1] - 1:
                test1a += 1
        if test1a == 4:
            Straight = True
            high = np.max(test1)
            return(Straight,high)
        else:
            return(False,0)
    
            

        
def FlushClass(cardval,suits): #returns true and highest flush card if player has a flush
    counts = np.unique(suits,return_counts=True)[1]
    if np.isin(5,counts):
        suitedorder = np.unique(suits,return_counts=True)[0] #1s,2c,3h,4d
        loc = np.where(counts == 5)[0][0] #used to find which type of flush it is 
        flush_type = suitedorder[loc]
        flush_cards = []
        for i in range(len(cardval)):
            if suits[i] == flush_type:
                flush_cards = np.append(flush_cards,cardval[i])
        high_card = np.max(flush_cards)
        
        return(True,high_card)
    else:
        return(False,0)
    

def CardVal(cards):
    "returns value of cards on a range of 0-12"
    #"0,1,2,3,4,5,6,7, 8, 9,10,11,12"
    #"2,3,4,5,6,7,8,9,10, J, Q, K, A"
    cardval = []
    whole_deck = np.arange(52)
    spades = np.split(whole_deck,4)[0]
    clubs = np.split(whole_deck,4)[1]
    hearts = np.split(whole_deck,4)[2]
    diamonds = np.split(whole_deck,4)[3]
    if isinstance(cards,int): #so i can find the value of a single card
        lookS = np.where(spades==cards)
        lookC = np.where(clubs==cards)
        lookH = np.where(hearts == cards)
        lookD = np.where(diamonds == cards)
        look = [lookS,lookC,lookH,lookD]
        for i in range(len(look)):
            x = len(look[i][0])
            if x == 1: return(x) #returns the card value for a single card
    for i in cards:
        if np.isin(i,spades):
            loc = np.where(i == spades)[0][0]
            cardval = np.append(cardval,loc) #
        elif np.isin(i,clubs):
            loc = np.where(i == clubs)[0][0]
            cardval = np.append(cardval,loc)
        elif np.isin(i,hearts):
            loc = np.where(i == hearts)[0][0]
            cardval = np.append(cardval,loc) 
        elif np.isin(i,diamonds):
            loc = np.where(i == diamonds)[0][0]
            cardval = np.append(cardval,loc)
    return(cardval)
            
def ShowCards(cards):
    show = ['2s','3s','4s','5s','6s','7s','8s','9s','10s','Js','Qs','Ks','As',
            '2c','3c','4c','5c','6c','7c','8c','9c','10c','Jc','Qc','Kc','Ac',
            '2h','3h','4h','5h','6h','7h','8h','9h','10h','Jh','Qh','Kh','Ah',
            '2d','3d','4d','5d','6d','7d','8d','9d','10d','Jd','Qd','Kd','Ad']
    ShowCards = []
    for i in range(len(cards)):
        ShowCards.append(show[int(cards[i])])
    return(ShowCards)
    
def ShowCard(card):
    show = ['2s','3s','4s','5s','6s','7s','8s','9s','10s','Js','Qs','Ks','As',
            '2c','3c','4c','5c','6c','7c','8c','9c','10c','Jc','Qc','Kc','Ac',
            '2h','3h','4h','5h','6h','7h','8h','9h','10h','Jh','Qh','Kh','Ah',
            '2d','3d','4d','5d','6d','7d','8d','9d','10d','Jd','Qd','Kd','Ad']
    ShowCard = show[card]
    return(ShowCard)

def find_vals(player_cards,flop,turn,river):
    p_cards = Cards(player_cards,flop,turn,river)
    
    p_cardvals = CardVal(p_cards)
    
    p_evaled = Evaluate(p_cardvals)
    
    p_suits = Suit(p_cards)
    
    p_suit_class = FlushClass(p_cardvals,p_suits)
    
    p_handval = Classify(p_cards,p_evaled,p_suit_class)
    return(p_handval)
    
def EndGame(stack):
    if stack == 0:
        return(False)
    else:
        return(True)

def create_input_state(hole_cards,flop,turn,river,move_p1,move_p2,cpos,hand_val): #need to include opponent's previous moves
    #if preflop, flop should equal 99
    #if preturn, turn should equal 99
    #if preriver, river should equal 99
    HC_vals = CardVal(hole_cards) # hole card value
    HC_suit = Suit(hole_cards)
    if isinstance(flop,int): 
        flop_vals = [0,0,0]
        flop_suit = [0,0,0]
    else:
        flop_vals = CardVal(flop)
        flop_suit = Suit(flop)
    if turn > 98: 
        turn_val = 0 #if the turn hasnt been reached yet
        turn_suit = 0
    else:
        turn_val = CardVal(turn)
        turn_suit = Suit(turn)
    if river > 98: 
        riv_val = 0
        riv_suit = 0
    else:
        riv_val = CardVal(river)
        riv_suit = Suit(river)
    cards_n_suit = [HC_vals[0],HC_vals[1],flop_vals[0],flop_vals[1],flop_vals[2],turn_val,riv_val,
             HC_suit[0],HC_suit[1],flop_suit[0],flop_suit[1],flop_suit[2],turn_suit,riv_suit]
    input_state = cards_n_suit
    #input_state.
    for i in range(16): input_state.append(move_p1[i])
    for i in range(16): input_state.append(move_p2[i])
    input_state.append(hand_val)
    return(input_state)

def prepare_move_state(previous_moves,last_move,PFTR): #PFTR is whether it is preflop postflop turn river
    #previous moves keeps getting appended based on moves in a specific betting round
    #     - should record the first 4 moves of the betting round, and should be equal to [] for first move
    # move state is the 16,1 list of all moves 
    if PFTR == 1: #preflop
        if len(np.where(np.asarray(previous_moves[:4])==0)[0]) == 0: return(previous_moves)
        where_pm_iszero = np.where(np.asarray(previous_moves[:4])==0)[0][0]
        previous_moves = previous_moves[:where_pm_iszero]
        if len(previous_moves) < 4:  previous_moves.append(last_move)
        FTR = [0] * int((4 - len(previous_moves) + 12)) # postflop turn river all equal to zero
        for i in range(len(FTR)): previous_moves.append(FTR[i])
        return(previous_moves)
    if PFTR == 2: #postflop
        if len(np.where(np.asarray(previous_moves[4:8])==0)[0]) == 0: return(previous_moves)
        where_pm_iszero = np.where(np.asarray(previous_moves[4:8])==0)[0][0] + 4
        previous_moves = previous_moves[:where_pm_iszero]
        if len(previous_moves) < 8: previous_moves.append(last_move)
        TR = [0] * int((8-len(previous_moves) + 8))
        for i in range(len(TR)): previous_moves.append(TR[i])
        return(previous_moves)
    if PFTR == 3: #post turn
        if len(np.where(np.asarray(previous_moves[8:12])==0)[0]) == 0: return(previous_moves)
        where_pm_iszero = np.where(np.asarray(previous_moves[8:12])==0)[0][0] + 8
        previous_moves = previous_moves[:where_pm_iszero]
        if len(previous_moves) < 12: previous_moves.append(last_move)
        T = [0] * int((12-len(previous_moves) + 4))
        for i in range(len(T)): previous_moves.append(T[i])
        return(previous_moves)
    if PFTR == 4: #post river
        if len(np.where(np.asarray(previous_moves[12:])==0)[0]) == 0: return(previous_moves)
        where_pm_iszero = np.where(np.asarray(previous_moves[12:])==0)[0][0] + 12
        previous_moves = previous_moves[:where_pm_iszero]
        if len(previous_moves) < 16: previous_moves.append(last_move)
        R = [0] * int((16-len(previous_moves)))
        for i in range(len(R)): previous_moves.append(R[i])
        return(previous_moves)

def prepare_replay_memory(old_input_state,new_input_state,action,reward,replay_memory):
    replay_memory.append([old_input_state,
                          action,
                          new_input_state,
                          reward])
    return(replay_memory)
#====================== end of functions to play game ============================================
#====================== functions to bet on hand =================================================

def Stack_Size(num_players,stack_size): #intializes a list of amounts to begin the game
    stack = []
    for i in range(num_players):
        stack = np.append(stack,stack_size) #each index is the stack for each player
    return(stack)
    
def UpdateStack(bet_amount,stack):
    stack -= bet_amount #takes bet amount out of stack size
    return(stack)
    
def Ante(stack,pot_size):
    stack -= big_blind
    pot_size += big_blind
    return(stack,pot_size)

def Check(bet_amout):
    bet_amount = 0
    return(bet_amount)

def Fold(p_stack,pot_size):
    p_stack += pot_size
    pot_size = 0
    return(p_stack,pot_size)


#====================== end functions to bet on hand =============================================
#====================== begin functions for GUI ==================================================
def p_move(player_turn,pot_size,p_stack,bet_amt,player):
    if player == 1:
        print('Player 1 Turn:')
    else:
        print('Player 2 Turn:')
        
    if player_turn == 0 and bet_amt == 0:
        move = int(input('Would you like to check(0), or raise(2)'))
    if player_turn == 1 and bet_amt == 0:
        move = int(input('Would you like to check(0) or raise(2)'))
    elif player_turn == 0 and bet_amt != 0:
        move = int(input('Would you like to  call(1), raise(2), fold(4)'))
    elif player_turn == 1 and bet_amt != 0:
        move = int(input('Would you like to  call(1), raise(2), fold(4)'))
    
    if move == 4: #fold
        move = -10
        if player == 1:
            print('Player 1 has folded')
        else:
            print('Player 2 has folded')
        return(p_stack,pot_size,bet_amt,True,False,move)
    elif move == 0: #check
        move = 5
        bet_amt = 0
        if player == 1 and player_turn == 0:
            print('Player 1 has checked')
            return(p_stack,pot_size,bet_amt,False,True,move)
        elif player == 1  and player_turn == 1:
            print('Player 1 has checked behind')
            return(p_stack,pot_size,bet_amt,False,False,move)
        elif player != 1 and player_turn == 0:
            print('Player 2 has checked')
            return(p_stack,pot_size,bet_amt,False,True,move)
        elif player != 1 and player_turn == 1:
            print('Player 2 has checked behind')
            return(p_stack,pot_size,bet_amt,False,False,move)  
    elif move == 1: #call
        move = 25
        pot_size += bet_amt
        p_stack -= bet_amt
        bet_amt = 0
        if player == 1:
            print('Player 1 has called')
        else:
            print('Player 2 has called')
        return(p_stack,pot_size,bet_amt,False,False,move)
    elif move == 2: #raise
        move = bet_amt
        bet_amt = int(input('Enter how much to raise:'))
        pot_size += bet_amt
        p_stack -= bet_amt
        if player == 1:
            print('Player 1 has raised ',bet_amt,'chips')
        else:
            print('Player 2 has raised ',bet_amt,'chips')
        return(p_stack,pot_size,bet_amt,False,True,move)
        
def comp_move(player_turn,pot_size,p_stack,bet_amt_i,player,input_state):
    #each move needs to be decided from comp_choice, but the computer cannot check if last player has raised
    three_bet = False
    if player_turn == 0 and bet_amt_i == 0: #beginning the action
        move, bet_amt = comp_choice(input_state,pot_size,p_stack,comp_network)
    if player_turn == 1 and bet_amt_i == 0: #in position, first move
        move, bet_amt = comp_choice(input_state,pot_size,p_stack,comp_network)
    elif player_turn == 0 and bet_amt_i != 0: #got raised
        move, bet_amt = comp_choice(input_state,pot_size,p_stack,comp_network)
        if bet_amt != 0: bet_amt, three_bet = bet_amt+bet_amt_i, True #3bet amount calls the previous bet and raises the desired amount
    elif player_turn == 1 and bet_amt_i != 0: #in position, got 3bet
        move, bet_amt = comp_choice(input_state,pot_size,p_stack,comp_network)
        if bet_amt != 0: bet_amt, three_bet = bet_amt+bet_amt_i, True
    
    if move == -10: #fold
        print('The computer has folded')
        return(p_stack,pot_size,bet_amt,True,False,move)
    elif move == 5: #check
        print('The computer has checked')
        bet_amt = 0
        if player == 1 and player_turn == 0:
            return(p_stack,pot_size,bet_amt,False,True,move)
        elif player == 1  and player_turn == 1:
            return(p_stack,pot_size,bet_amt,False,False,move)
        elif player != 1 and player_turn == 0:
            return(p_stack,pot_size,bet_amt,False,True,move)
        elif player != 1 and player_turn == 1:
            return(p_stack,pot_size,bet_amt,False,False,move)  
    elif move == 25: #call
        print('The computer has called')
        pot_size += bet_amt
        p_stack -= bet_amt
        bet_amt = 0
        return(p_stack,pot_size,bet_amt,False,False,move)
    elif three_bet:
        pot_size += bet_amt
        p_stack -= bet_amt
        bet_amt -= bet_amt_i
        print('The computer has called',bet_amt_i,'chips and raised',bet_amt,'chips')
        return(p_stack,pot_size,bet_amt,False,True,move)
    else: #raise
        print('The computer has raised',bet_amt,'chips')
        pot_size += bet_amt
        p_stack -= bet_amt
        return(p_stack,pot_size,bet_amt,False,True,move)

def comp_choice(input_state,pot_size,stack_size,comp_network):
    global epsilon
    global max_ep
    global min_ep
    global hands_played
    global decay
    rand = np.random.uniform(0,1)
    if rand <= epsilon: #explore
        act = np.random.randint(0,8)
    else: #exploit
        act = np.argmax(comp_network.predict(input_state))
    
    if act == 0: #fold
        move = -10
        bet_amt = 0
    elif act == 1: #check
        move = 5
        bet_amt = 0
    elif act == 2: #call
        move = 25
        bet_amt = 0
    elif act == 3: # raise 1/4 pot
        move = 125
        bet_amt = 0.25 * pot_size
    elif act == 4: # raise 1/2 pot
        move = 145
        bet_amt = 0.5 * pot_size
    elif act == 5: #raise 3/4 pot
        move = 165
        bet_amt = 0.75 * pot_size
    elif act == 6: #raise pot
        move = 185
        bet_amt = pot_size
    elif act == 7: # raise 5/4 pot
        move = 205
        bet_amt = 1.25 * pot_size
    elif act == 8: # all in 
        move = 250
        bet_amt = stack_size
    if bet_amt > stack_size:
        bet_amt = stack_size
    
    #epsilon = min_ep + (max_ep - min_ep) * np.exp(-decay*hands_played)
    return(move,bet_amt)
#======================= end function for GUI =======================================================
#======================= begin functions for gameplay ===============================================



p1_stack = Stack_Size(num_players,stack_size)[0]
p2_stack = Stack_Size(num_players,stack_size)[1]

def human_vs_human(play,p1_stack,p2_stack):
    
    hands_played = 0
    
    while play:
        bet_amt = 0
        deck = np.arange(52)
        #assigning hole cards and removing them from the deck. 
        deck, hole_cards = DealCards(deck,num_players) #gives players random hole cards and removes them from the deck
        p1_hand = hole_cards[0] #assigns p1 hole cards
        p2_hand = hole_cards[1] #assigns p2 hole cards
        play = False
        #each player must pay an ante.
        pot_size = 0
        p1_stack, pot_size = Ante(p1_stack,pot_size) #updates p1 and p2 stack size and pot size
        p2_stack, pot_size = Ante(p2_stack,pot_size)
        p1_cont, p2_cont = False, False
        #---------- preflop betting action  ------------------
        print('\n')
        print('*****************  Pre-flop betting  ***************************')
        action = True
        while action:
            if (hands_played % 2) == 0: #player 1 goes first
                p1_stack, pot_size, bet_amt, p1_cont,action = move(0,pot_size,p1_stack,bet_amt,1)
                if action: p2_stack, pot_size, bet_amt, p2_cont,action  = move(1,pot_size,p2_stack,bet_amt,2)
            elif (hands_played % 2) == 1: #player 2 goes first
                player2_turn = 0
                p2_stack, pot_size, bet_amt, p2_cont,action  = move(0,pot_size,p2_stack,bet_amt,2)
                if action: p1_stack, pot_size, bet_amt, p1_cont,action = move(1,pot_size,p1_stack,bet_amt,1)
        #------------ end action ------------------
        # if a player folded, game needs to end
        if p2_cont == True: #if player 2 folded
            p1_stack, pot_size = Fold(p1_stack,pot_size)
            continue
        if p1_cont == True: #if p1 folded
            p2_stack, pot_size = Fold(p2_stack,pot_size)
            continue
        #  ------------------------------------------
    
        # ---------- showing cards ---------------------------
        deck, flop = Flop(deck)
        showflop = ShowCards(flop)
        print('====================================================================')
        print('========== Betting action is over.The Flop is shown below ==========')
        print('=================     ',showflop,'     =====================')
        print('====================================================================')
        # --------------------------------------------------
        #
        print('\n')
        print('********************  Post-flop betting  ***************************')
        action = True
        while action:
            if (hands_played % 2) == 0: #player 1 goes first
                p1_stack, pot_size, bet_amt, p1_cont,action = move(0,pot_size,p1_stack,bet_amt,1)
                if action: p2_stack, pot_size, bet_amt, p2_cont,action  = move(1,pot_size,p2_stack,bet_amt,2)
            elif (hands_played % 2) == 1: #player 2 goes first
                player2_turn = 0
                p2_stack, pot_size, bet_amt, p2_cont,action  = move(0,pot_size,p2_stack,bet_amt,2)
                if action: p1_stack, pot_size, bet_amt, p1_cont,action = move(1,pot_size,p1_stack,bet_amt,1)
        #------------ end action ------------------
        # if a player folded, game needs to end
        if p2_cont == True: #if player 2 folded
            p1_stack, pot_size = Fold(p1_stack,pot_size)
            continue
        if p1_cont == True: #if p1 folded
            p2_stack, pot_size = Fold(p2_stack,pot_size)
            continue
        #----------------------------------------------------
        #----------- display board after turn ---------------
        deck, turn = TurnRiver(deck)
        showturn = ShowCard(turn)
        print('====================================================================')
        print('========== Betting action is over.The board is shown below =========')
        print('==============     ',showflop,[showturn],'     =================')
        print('====================================================================')    
        #------------------------------------------------------
        print('\n')
        print('******************  Post-turn betting  ***************************')
        action = True
        while action:
            if (hands_played % 2) == 0: #player 1 goes first
                p1_stack, pot_size, bet_amt, p1_cont,action = move(0,pot_size,p1_stack,bet_amt,1)
                if action: p2_stack, pot_size, bet_amt, p2_cont,action  = move(1,pot_size,p2_stack,bet_amt,2)
            elif (hands_played % 2) == 1: #player 2 goes first
                player2_turn = 0
                p2_stack, pot_size, bet_amt, p2_cont,action  = move(0,pot_size,p2_stack,bet_amt,2)
                if action: p1_stack, pot_size, bet_amt, p1_cont,action = move(1,pot_size,p1_stack,bet_amt,1)
        #------------ end action ------------------
        # if a player folded, game needs to end
        if p2_cont == True: #if player 2 folded
            p1_stack, pot_size = Fold(p1_stack,pot_size)
            continue
        if p1_cont == True: #if p1 folded
            p2_stack, pot_size = Fold(p2_stack,pot_size)
            continue
        #--------------------------------------------------------
        #----------- display board after river ---------------
        deck, river = TurnRiver(deck)
        showriver = ShowCard(river)
        print('====================================================================')
        print('========== Betting action is over.The board is shown below =========')
        print('============     ',showflop,[showturn],[showriver],'     ============')
        print('====================================================================')    
        #------------------------------------------------------
        #------------------------------------------------------
        print('\n')
        print('******************  Post-river betting  ***************************')
        action = True
        while action:
            if (hands_played % 2) == 0: #player 1 goes first
                p1_stack, pot_size, bet_amt, p1_cont,action = move(0,pot_size,p1_stack,bet_amt,1)
                if action: p2_stack, pot_size, bet_amt, p2_cont,action  = move(1,pot_size,p2_stack,bet_amt,2)
            elif (hands_played % 2) == 1: #player 2 goes first
                player2_turn = 0
                p2_stack, pot_size, bet_amt, p2_cont,action  = move(0,pot_size,p2_stack,bet_amt,2)
                if action: p1_stack, pot_size, bet_amt, p1_cont,action = move(1,pot_size,p1_stack,bet_amt,1)
                #------------ end action ------------------
                # if a player folded, game needs to end
        if p2_cont == True: #if player 2 folded
            p1_stack, pot_size = Fold(p1_stack,pot_size)
            continue
        if p1_cont == True: #if p1 folded
            p2_stack, pot_size = Fold(p2_stack,pot_size)
            continue
        #--------------------------------------------------------
        # now, decides who is the winner
        p1_handval = find_vals(p1_hand,flop,turn,river)
        p2_handval = find_vals(p2_hand,flop,turn,river)
        print('\n')
        if p1_handval > p2_handval:
            print('Player 1 has won the hand!')
            p1_stack += pot_size
            potsize = 0
        elif p2_handval > p1_handval:
            print('Player 2 has won the hand!')
            p2_stack += pot_size
            potsize = 0
        elif p1_handval == p2_handval:
            print('The hand resulted in a tie')
            p1_stack += (pot_size/2)
            p2_stack += (pot_size/2)
            pot_size = 0
        
    
    
        hands_played += 1
        x = 0
        play = EndGame(x) #checks to see if the game is over(by seeing if they have no more chips)


def comp_vs_human(comp_play,p1_stack,p2_stack):
    hands_played = 0
    replay_memory_comp = deque(maxlen=50_000)
    
    while comp_play:
        #------------------------- assigning cards and taking ante -----------------------
        bet_amt = 0
        deck = np.arange(52)
        #assigning hole cards and removing them from the deck. 
        deck, hole_cards = DealCards(deck,num_players) #gives players random hole cards and removes them from the deck
        p1_hand = hole_cards[0] #assigns p1 hole cards
        p2_hand = hole_cards[1] #assigns p2 hole cards
        comp_play = False
        #each player must pay an ante.
        pot_size = 0
        p1_stack, pot_size = Ante(p1_stack,pot_size) #updates p1 and p2 stack size and pot size
        p2_stack, pot_size = Ante(p2_stack,pot_size)
        
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #  move_state_c1 contains all of the computer's moves
        #  move_state_c2 contains all of the human's moves
        #------------------------------------------------------------------------------------
        #------------------------- preflop action -------------------------------------------
        print('\n')
        print('*****************  Pre-flop betting  ***************************')
        print('==== Hole Cards:',ShowCards(p1_hand),' || Pot size:',pot_size,'  ===')
        action, count = True, 0
        comp_handval = find_vals(p2_hand,99,99,99) #computer is p2
        while action:
            if (hands_played % 2) == 0: #player 1 goes first
                c1_pos = 10
                c2_pos = -10
                if count==0: move_state_c1, move_state_c2 = [0]*16, [0]*16 #preparing move state at start of action
                p1_stack, pot_size, bet_amt, p1_cont,action, move = p_move(0,pot_size,p1_stack,bet_amt,1) #c1 move
                move_state_c2 = prepare_move_state(move_state_c2,move,1)
                if action: 
                    input_state = create_input_state(p2_hand,99,99,99,move_state_c2,move_state_c1,c2_pos,comp_handval)
                    p2_stack, pot_size, bet_amt, p2_cont,action, move  = comp_move(1,pot_size,p2_stack,bet_amt,2,input_state) #c2 move
                    move_state_c1 = prepare_move_state(move_state_c1,move,1)
                count+= 1
            elif (hands_played % 2) == 1: #player 2 goes first
                player2_turn = 0
                c2_pos = 10
                c1_pos = -10
                if count==0: move_state_c2, move_state_c1 = [0]*16, [0]*16
                input_state = create_input_state(p2_hand,99,99,99,move_state_c2,move_state_c1,c2_pos,comp_handval)
                p2_stack, pot_size, bet_amt, p2_cont,action,move  = comp_move(0,pot_size,p2_stack,bet_amt,2,input_state)
                move_state_c1 = prepare_move_state(move_state_c1,move,1)
                if action: 
                    p1_stack, pot_size, bet_amt, p1_cont,action,move = p_move(0,pot_size,p1_stack,bet_amt,1)
                    move_state_c2 = prepare_move_state(move_state_c2,move,1)
                count+=1
        #------------ end action ------------------
        # if a player folded, game needs to end
        if p2_cont == True: #if player 2 folded
            p1_stack, pot_size = Fold(p1_stack,pot_size)
            continue
        if p1_cont == True: #if p1 folded
            p2_stack, pot_size = Fold(p2_stack,pot_size)
            continue
        
        #------------------------ end preflop action -----------------------------------------
        #------------------------ flop
        deck, flop = Flop(deck)
        showflop = ShowCards(flop)
        print('====================================================================')
        print('========== Betting action is over.The Flop is shown below ==========')
        print('=================     ',showflop,'     =====================')
        print('====================================================================')
        print('==== Hole Cards:',ShowCards(p1_hand),' || Pot size:',pot_size,'  ===')
        #------------------------
        print('\n')
        print('********************  Post-flop betting  ***************************')
        #------------------------- postflop action ----------------------------------------------
        action = True
        comp_handval = find_vals(p2_hand,flop,99,99) #computer is p2
        while action:
            if (hands_played % 2) == 0: #player 1 goes first
                p1_stack, pot_size, bet_amt, p1_cont,action, move = p_move(0,pot_size,p1_stack,bet_amt,1) #c1 move
                move_state_c2 = prepare_move_state(move_state_c2,move,2)
                if action: 
                    input_state = create_input_state(p2_hand,flop,99,99,move_state_c2,move_state_c1,c2_pos,comp_handval)
                    p2_stack, pot_size, bet_amt, p2_cont,action, move  = comp_move(1,pot_size,p2_stack,bet_amt,2,input_state) #c2 move
                    move_state_c1 = prepare_move_state(move_state_c1,move,2)
                count+= 1
            elif (hands_played % 2) == 1: #player 2 goes first
                player2_turn = 0
                input_state = create_input_state(p2_hand,flop,99,99,move_state_c2,move_state_c1,c2_pos,comp_handval)
                p2_stack, pot_size, bet_amt, p2_cont,action,move  = comp_move(0,pot_size,p2_stack,bet_amt,2,input_state)
                move_state_c1 = prepare_move_state(move_state_c1,move,2)
                if action: 
                    p1_stack, pot_size, bet_amt, p1_cont,action,move = p_move(0,pot_size,p1_stack,bet_amt,1)
                    move_state_c2 = prepare_move_state(move_state_c2,move,2)
        #------------ end action ------------------
        # if a player folded, game needs to end
        if p2_cont == True: #if player 2 folded
            p1_stack, pot_size = Fold(p1_stack,pot_size)
            continue
        if p1_cont == True: #if p1 folded
            p2_stack, pot_size = Fold(p2_stack,pot_size)
            continue
        #------------------------ turn
        deck, turn = TurnRiver(deck)
        turn = int(turn)
        showturn = ShowCard(turn)
        print('====================================================================')
        print('========== Betting action is over.The board is shown below =========')
        print('==============     ',showflop,[showturn],'     =================')
        print('====================================================================') 
        print('==== Hole Cards:',ShowCards(p1_hand),' || Pot size:',pot_size,'  ===')
        #------------------------------------------------------
        print('\n')
        print('******************  Post-turn betting  ***************************')
        #------------------------
        #------------------------ post turn action --------------------------------------------

        action= True
        comp_handval = find_vals(p2_hand,flop,turn,99) #computer is p2
        while action:
            if (hands_played % 2) == 0: #player 1 goes first
                p1_stack, pot_size, bet_amt, p1_cont,action, move = p_move(0,pot_size,p1_stack,bet_amt,1) #c1 move
                move_state_c2 = prepare_move_state(move_state_c2,move,3)
                if action: 
                    input_state = create_input_state(p2_hand,flop,turn,99,move_state_c2,move_state_c1,c2_pos,comp_handval)
                    p2_stack, pot_size, bet_amt, p2_cont,action, move  = comp_move(1,pot_size,p2_stack,bet_amt,2,input_state) #c2 move
                    move_state_c1 = prepare_move_state(move_state_c1,move,3)
            elif (hands_played % 2) == 1: #player 2 goes first
                player2_turn = 0
                input_state = create_input_state(p2_hand,flop,turn,99,move_state_c2,move_state_c1,c2_pos,comp_handval)
                p2_stack, pot_size, bet_amt, p2_cont,action,move  = comp_move(0,pot_size,p2_stack,bet_amt,2,input_state)
                move_state_c1 = prepare_move_state(move_state_c1,move,3)
                if action: 
                    p1_stack, pot_size, bet_amt, p1_cont,action,move = p_move(0,pot_size,p1_stack,bet_amt,1)
                    move_state_c2 = prepare_move_state(move_state_c2,move,3)
        #------------ end action ------------------
        #--------------------------
        if p2_cont == True: #if player 2 folded
            p1_stack, pot_size = Fold(p1_stack,pot_size)
            continue
        if p1_cont == True: #if p1 folded
            p2_stack, pot_size = Fold(p2_stack,pot_size)
            continue
        #--------------------------------------------------------
        #----------- river ---------------
        deck, river = TurnRiver(deck)
        river = int(river)
        showriver = ShowCard(river)
        print('====================================================================')
        print('========== Betting action is over.The board is shown below =========')
        print('============     ',showflop,[showturn],[showriver],'     ============')
        print('====================================================================')   
        print('======= Hole Cards:',ShowCards(p1_hand),'----- Pot size:',pot_size,' ===========')
        print('====================================================================')
        #------------------------------------------------------
        #------------------------------------------------------
        print('\n')
        print('******************  Post-river betting  ***************************')
        #--------------------------------
        #-------------------------------post river action -------------------------------------
        action, count = True, 0
        comp_handval = find_vals(p2_hand,flop,turn,river) #computer is p2
        while action:
            if (hands_played % 2) == 0: #player 1 goes first
                p1_stack, pot_size, bet_amt, p1_cont,action, move = p_move(0,pot_size,p1_stack,bet_amt,1) #c1 move
                move_state_c2 = prepare_move_state(move_state_c2,move,4)
                if action: 
                    input_state = create_input_state(p2_hand,flop,turn,river,move_state_c2,move_state_c1,c2_pos,comp_handval)
                    p2_stack, pot_size, bet_amt, p2_cont,action, move  = comp_move(1,pot_size,p2_stack,bet_amt,2,input_state) #c2 move
                    move_state_c1 = prepare_move_state(move_state_c1,move,4)
            elif (hands_played % 2) == 1: #player 2 goes first
                input_state = create_input_state(p2_hand,flop,turn,river,move_state_c2,move_state_c1,c2_pos,comp_handval)
                p2_stack, pot_size, bet_amt, p2_cont,action,move  = comp_move(0,pot_size,p2_stack,bet_amt,2,input_state)
                move_state_c1 = prepare_move_state(move_state_c1,move,4)
                if action: 
                    p1_stack, pot_size, bet_amt, p1_cont,action,move = p_move(0,pot_size,p1_stack,bet_amt,1)
                    move_state_c2 = prepare_move_state(move_state_c2,move,4)
        #------------ end action ------------------
        #-------------------------------------
        if p2_cont == True: #if player 2 folded
            p1_stack, pot_size = Fold(p1_stack,pot_size)
            continue
        if p1_cont == True: #if p1 folded
            p2_stack, pot_size = Fold(p2_stack,pot_size)
            continue
        #-------------- finding who won --------------------------------------------------------------------------
        p1_handval = find_vals(p1_hand,flop,turn,river) #human is p1
        p2_handval = find_vals(p2_hand,flop,turn,river) #computer is p2
        if p1_handval > p2_handval:
            print('You have won the hand with:',ShowCards(p1_hand))
            print('The computer had:',ShowCards(p2_hand))
            print('The pot won totals',pot_size)
            p1_stack += pot_size
            print('Your stack is now',p1_stack,'chips')
            potsize = 0
        elif p2_handval > p1_handval:
            print('The computer has won the hand with:',ShowCards(p2_hand))
            print('The pot won totals',pot_size)
            p2_stack += pot_size
            print('The computers stack is now',p2_stack,'chips')
            potsize = 0
        elif p1_handval == p2_handval:
            print('The hand has ended in a tie')
            print('Both players receive',pot_size/2,'chips')
            p1_stack += (pot_size/2)
            p2_stack += (pot_size/2)
            pot_size = 0

        hands_played += 1
        if hands_played == 2: break
    return(input_state)
        
        