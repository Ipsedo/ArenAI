APP_MODULES := bullet phyvr libpng
APP_ABI := all
APP_OPTIM := release

#We only need STL for placement new (#include <new>) 
#We don't use STL in Bullet
APP_STL                 := c++_shared