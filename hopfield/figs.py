import numpy

fig1 =  ['                                ',
         '   ***************              ',
         '    *             *             ',
         '     *             *            ',
         '     *             *            ',
         '      *             *           ',
         '      *             *           ',
         '       *            *           ',
         '       *             *          ',
         '       *             ****       ',
         '        *            **  **     ',
         '    *** *            ***   *    ',
         '  **   **            *****  *   ',
         ' *       *         *****     *  ',
         '*         ***  *****         *  ',
         '*           ****              * ',
         '*                             * ',
         '*                          **** ',
         '*                       ***   * ',
         '***                  ***      * ',
         '*  **             ***         * ',
         '*    **        ***            * ',
         '*      **   ***         ***   * ',
         '*        ***         ***   ***  ',
         '*                 ***         **',
         ' **            ***              ',
         '   **         **                ',
         '     **       **               *',
         '       **   ****            *** ',
         '         ***    **        **    ',
         '                  **   ***      ',
         '                    ***         ']

fig2 =  ['              *****         *** ',
         '             *     *       *   *',
         '             *     *      *     ',
         '      **     *     *     *      ',
         '     *  *  ***     ***  *       ',
         '    *    **  *     *****       *',
         '   *         *     ****       * ',
         '   *       ***     ***       *  ',
         '    *     *  *     **       *   ',
         '     *  **   *     *       *    ',
         '     *  *    *            **    ',
         '    *  *     *           ****   ',
         '    *  *     *           ****   ',
         ' ***  *      *            ** ***',
         '*     *      *     *       *    ',
         '*     *      *     **       **  ',
         '*     *      *     ***       ** ',
         '*     *      *     ****       **',
         '*     *      *     *****       *',
         ' ***  *      *     ******       ',
         '    *  *     *     *******      ',
         '    ** *     *     ****** *     ',
         '     *  *    *     *****  **   *',
         '     *   *   ***********   **** ',
         '    *     *  *********      *   ',
         '   *       *********         *  ',
         '   **                        *  ',
         '    **   **           **    *   ',
         '     ** *  **       **  *  *    ',
         '      **     *     *     **     ',
         '             *     *            ',
         '             *    **            ']

fig3 =  ['********************************',
         '********************************',
         '********************************',
         '********************************',
         '****** *************************',
         '******* ************************',
         '******** * *********************',
         '*** *****  *********************',
         '**** ***   *********************',
         '***** * *** *     **************',
         '******  ***  *****  ************',
         '*****   ** ********* ***********',
         '********  *********** **********',
         '******** *** ****** ** *********',
         '******** ***   **** ** *********',
         '******* ****    *** *** ********',
         '******* ****      * *** ********',
         '*                              *',
         '******* ****      * *** ********',
         '******* ****    *** *** ********',
         '******** ***   **** ** *********',
         '******** *** ****** ** *********',
         '********* *********** **********',
         '********** ********* ***********',
         '***********  *****  ************',
         '*************     **************',
         '********************************',
         '********************************',
         '********************************',
         '********************************',
         '********************************',
         '********************************']

fig4 =  ['********************************',
         '*********************** ********',
         '*********************** ********',
         '*********************** ********',
         '*********************** ********',
         '*********************** ********',
         '***************     *** ********',
         '*************  *****  * ********',
         '***********  ****  ***  ********',
         '********** ******  ****  *******',
         '********* *******       * ******',
         '********* *******  ****** ******',
         '******** ********  ******* *****',
         '******** ****  *********** *****',
         '******* *****  **  *** **** ****',
         '******* *****  **  ***  *** ****',
         '*              **            ***',
         '******* *****  **  ***  *** ****',
         '******* *****  **  *** **** ****',
         '******** ****  *********** *****',
         '******** ********  ******* *****',
         '********* *******  ****** ******',
         '********* *******       * ******',
         '********** ******  ****  *******',
         '***********  ****  ***  ********',
         '*************  *****  * ********',
         '***************     *** ********',
         '*********************** ********',
         '*********************** ********',
         '*********************** ********',
         '*********************** ********',
         '********************************']

fig5 =  ['********************************',
         '********************************',
         '********************************',
         '********************************',
         '*******                     ****',
         '****** ********************* ***',
         '***** *********************** **',
         '***** ***                 *** **',
         '***** ** ***************** ** **',
         '***** ** ***************** ** **',
         '***** ** ***************** ** **',
         '***** ** **** *** ** ***** ** **',
         '***** ** **** *** ** ***** ** **',
         '***** ** ******** ******** ** **',
         '***** ** ******** ******** ** **',
         '***** ** *******  ******** ** **',
         '***** ** ***************** ** **',
         '***** ** ***** **** ****** ** **',
         '***** ** ******    ******* ** **',
         '***** ** ***************** ** **',
         '***** ** ***************** ** **',
         '***** ***                 *** **',
         '***** *********************** **',
         '***** *********************** **',
         '***** *********************** **',
         '***** *********************** **',
         '***** **************      *** **',
         '***** **  ******************* **',
         '***** *********************** **',
         '***** *********************** **',
         '***** *********************** **',
         '******                       ***']

fig6 =  ['*****************   ************',
         '*******     ***  * *************',
         '******    **   ** **************',
         '***** *   *****  ***************',
         '**** ** ******  ****************',
         '**** * **  ** ** ***************',
         '****  ***  ** **  **************',
         '****  ****** ***   *************',
         '****  ****** ** ** *************',
         '***** *****   ***** ************',
         '*****  **    ****** **** *******',
         '******    **  ****** ** ********',
         '*******  ***  ****** **  *******',
         '*******  ***  ****** ** ********',
         '*******  ***  ******   * *******',
         '*******  ***  ****** ***********',
         '*******  ***  ****** *** *******',
         '*******  ***  ***** *** ********',
         '*******  **  ******      *******',
         '******** **  ****** *** ********',
         '******** * ******* ***** *******',
         '********  ** ***** *************',
         '********* *   **** *************',
         '*********   *  ***  ************',
         '********** **   *   ************',
         '********** *  * * *  ***********',
         '*********** * ** * * ***********',
         '*********** * *** * * **********',
         '************  **** *  **********',
         '************  ***** *  *********',
         '************* ****** * *********',
         '*********************  *********']

fig7 =  ['********************************',
         '*********    *******************',
         '*******    **  *****************',
         '******  ******* ****************',
         '*****  ******** ****************',
         '***** ***  ****    *************',
         '****  ***  ****  *******  ******',
         '**** ********** ******* *  *****',
         '**** ********* ******** *  *****',
         '***  ********* ******* *  ******',
         '*** **********  ****** *  ******',
         '*** *       **** **** *  *******',
         '*** *          ** *** *  *******',
         '*** **       ***** * *  ********',
         '*** ***     *******  *  ********',
         '*** ****   *********   *********',
         '*** ***** ***********  *********',
         '**** ***************** *********',
         '**** ****************** ********',
         '***** *************   ** *******',
         '****** *********** ***   *******',
         '*******  *******  **************',
         '*********       ****************',
         '********* ***** ****************',
         '********* ***** ****************',
         '******** ***** *****************',
         '******** ***** *****************',
         '*****   * *   * *************** ',
         '********************************',
         '********************************',
         '********************************',
         '********************************']

fig8 =  ['****************     ***********',
         '***************       **********',
         '**************         *********',
         '**************         *********',
         '************** *  **   *********',
         '***************   * *  *********',
         '**************  **  *  *********',
         '******   *****  *****  *********',
         '******* * **** ****    *********',
         '**** ***  ****     **   ********',
         '****  * * **** ** ****   *******',
         '**** * ** *** ********    ******',
         '***** ** ***   ******     ******',
         '****** * ***   ******      *****',
         '****** * **    ******       ****',
         '****** *       ******       ****',
         '*****  *                    ****',
         '*****         *      *       ***',
         '*****                        ***',
         '****** * **                  ***',
         '****** * **                  ***',
         '****** * **                 ****',
         '****** * **                 ****',
         '****** ** *                *****',
         '***** ** *               *******',
         '***** * *  *             *******',
         '*****  *** *            ********',
         '***** * *****           ********',
         '******   ****    ***    ********',
         '*********    **** * ****    ****',
         '********  ***    ***    ***  ***',
         '**********   ***********   *****']

fig9 =  ['*****                        ***',
         '***** ********************** ***',
         '***** ****   ****       **** ***',
         '***** *** *** *** ***** **** ***',
         '*****     ***     *****      ***',
         '***** *** *** * * ***** * ** ***',
         '***** ****   ** *       * ** ***',
         '***** ********* ********* ** ***',
         '***** ********* ***   *** ** ***',
         '***** ********* ** *** ** ** ***',
         '***** **********   ***   *** ***',
         '***** ************ *** ***** ***',
         '***** *************   ****** ***',
         '***** ********************** ***',
         '***** ********************** ***',
         '***                           **',
         '** *************************** *',
         '** ** * * * * * * * * * * * ** *',
         '** ****     ********     ***** *',
         '** **  * * * * * *  * * *  *** *',
         '** * * * * * ****** * * * * ** *',
         '** * * * * *    *   * * * * ** *',
         '** * * ***** * ** * ***** * ** *',
         '** * ******* * *  * ******* ** *',
         '** * ********* ** ********* ** *',
         '** * ********* ** ********* ** *',
         '***  *********    *********   **',
         '***** ******* **** ******* *****',
         '***** ****** ****** ****** *****',
         '***** ***** ******** ***** *****',
         '**** ****** ******** ****** ****',
         '****        ********        ****']

fig11 = [' *******  * *                   ',
         ' ***  *  *   *  **              ',
         '*** ****   *** *  *             ',
         '*** *   *  ****    *            ',
         ' **  * * *   *     *            ',
         '* * **    *  *      *           ',
         '* *** *** * * *     *           ',
         '    *  * ****       *           ',
         '*  ****  *  ****     *          ',
         '** * * *  *  *       ****       ',
         '   * *** ***** *     **  **     ',
         '**     **** * **     ***   *    ',
         '  *   *   *          *****  *   ',
         '***    * **  **    *****     *  ',
         ' ** *  *   *    ****         *  ',
         '*   * * *  *   *              * ',
         '* ** * * *                    * ',
         '* **  *   * **             **** ',
         '  *  *   *****          ***   * ',
         ' *  *  *   ** *      ***      * ',
         '***    *   * **   ***         * ',
         ' **  **  *  ** ***            * ',
         '**   * ***  **          ***   * ',
         '*  * **  ** * *      ***   ***  ',
         ' **** ***  **     ***         **',
         '*  *     *     ***              ',
         '* *  * * *  * *                 ',
         '  *    **   * **               *',
         ' *     **  **  *            *** ',
         '*   ** ** **  ****        **    ',
         ' **  ** * ***  *  **   ***      ',
         '*  * *  *** ***     ***         ']

fig22 = ['*******************         *** ',
         '****************   *       *   *',
         '****************   *      *     ',
         '****************   *     *      ',
         '****** *********   ***  *       ',
         '******* ********   *****       *',
         '******** * *****   ****       * ',
         '*** *****  *****   ***       *  ',
         '**** ***   *****   **       *   ',
         '***** * *** *      *       *    ',
         '******  ***  ***          **    ',
         '*****   ** *****         ****   ',
         '********  ******         ****   ',
         '******** *** ***          ** ***',
         '******** ***   *   *       *    ',
         '******* ****       **       **  ',
         '******* ****       ***       ** ',
         '*                  ****       **',
         '******* ****       *****       *',
         '******* ****       ******       ',
         '******** ***   *   *******      ',
         '******** *** ***   ****** *     ',
         '********* ******   *****  **   *',
         '********** *************   **** ',
         '***********  *********      *   ',
         '*************   ****         *  ',
         '****************             *  ',
         '****************      **    *   ',
         '****************    **  *  *    ',
         '****************   *     **     ',
         '****************   *            ',
         '****************  **            ']


def makePattern(fig):
    return numpy.array([[-1 if c==' ' else 1 for c in row] for row in fig]).reshape((1024))

p1 = makePattern(fig1)
p2 = makePattern(fig2)
p3 = makePattern(fig3)
p4 = makePattern(fig4)
p5 = makePattern(fig5)
p6 = makePattern(fig6)
p7 = makePattern(fig7)
p8 = makePattern(fig8)
p9 = makePattern(fig9)

p11 = makePattern(fig11)
p22 = makePattern(fig22)