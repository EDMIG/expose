
from decimal import *
getcontext().prec=34
x00=Decimal('1.4491376746189438880169333150257')
x01=Decimal('1.7320508075688772935274463415058')
y00=Decimal('-0.50484610459985752828933163777342')
y10=Decimal('-0.57482394653326920224545723244855')

z00=Decimal('-1.7272157908631477151639325868808')

t=x00*y00+x01*y10

tt=x00/x01+y00/y10+(Decimal('10')*z00).exp()

print t
print tt
