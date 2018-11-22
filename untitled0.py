# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 21:26:55 2018

@author: User
"""
"""
a = int(input ("Payment :"))
b = int(input ("Price :"))
x = a-b # x = change
if x < 0 :
        print ('not enough')
elif x == 0 :
        print ('thank you')
else :
    num_fifty = x // 50
    x -= 50 * num_fifty
    num_ten = x // 10
    x -= 10 * num_ten
    num_five = x // 5
    x -= 5 * num_five  
    num_one = x // 1
    print ("\nCombine by :\n",num_fifty,"fifity\n",num_ten,"ten\n",num_five,"five\n",num_one,"one coin")
"""    
x = int(input("Input the price: "))
y = int(input("Input the payment: "))
change = y-x
fifty = change//50
ten = (change-fifty*50)//10
five = (change-fifty*50-ten*10)//5
one = (change-fifty*50-ten*10-five*5)//1
print('The change will be:',change)
print("Combined by: \n50 *", fifty, "\n10 *", ten, "\n5 *", five, "\n1 *", one)
#""裡面是字串，字串之間的, fifty,  逗號是連接字串跟fifty這個參數   
#"\n10 *"   \n是跳脫字元，代表換行  因此意思是先換行再印出10

#print(fifty/n)  #  跑不出來
print(fifty,"\n")  #  跑得出來，fifty這個參數內的值印出來後，換行
#print(/n*)  #  跑不出來
print("\n*")  #   先換行後再印出*