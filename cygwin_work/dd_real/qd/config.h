#pragma once

#ifndef QD_API
#define QD_API /**/
#endif

/* Set the following to 1 to use slightly inaccurate but faster
version of multiplication. */
#ifndef QD_SLOPPY_MUL
#define QD_SLOPPY_MUL 1
#endif

/* Set the following to 1 to use slightly inaccurate but faster
version of division. */
#ifndef QD_SLOPPY_DIV
#define QD_SLOPPY_DIV 1
#endif

/* Define this macro to be the isfinite(x) function. */
#ifndef QD_ISFINITE
#define QD_ISFINITE(x) std::isfinite(x)
#endif

/* Define this macro to be the isinf(x) function. */
#ifndef QD_ISINF
#define QD_ISINF(x) std::isinf(x)
#endif

/* Define this macro to be the isnan(x) function. */
#ifndef QD_ISNAN
#define QD_ISNAN(x) std::isnan(x)
#endif

/*
20180710,CXG
ʹ��fma & fmsָ����٣�
*/

//�������������
//#define QD_FMS(a,b,c) ((a)*(b)-c)
//#define QD_FMA(a,b,c) ((a)*(b)+c)

//��������ȷ
//������������ߣ�30%����
#define QD_FMS(a,b,c) fma(a,b,-c)
#define QD_FMA(a,b,c) fma(a,b, c)




