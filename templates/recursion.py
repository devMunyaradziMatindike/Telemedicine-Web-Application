#def look_for_key(box):
#    for item in box:
#        if item.is_a_box():
#            look_for_key(item) 
#        elif item.is_a_key():
#          print( "found the key!")

#def countdown(i):
#    print(i)
#    if i<=0:
#        return #base case
#    else:
#        countdown(i-1) #recursive case
#countdown(-1)

# call stack 
def greet(name):
    print("hello ,"+name+ "!")
    greet2(name)
    print("Getting ready to say bye")
    bye()

def greet2(name):
    print("how are you"+name+"?")

def bye():
    print("Ok bye.")    

greet("Munyaradzi")    