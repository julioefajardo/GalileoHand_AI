/* Universidad Galileo
 * Alan Turing Research Lab
 * Julio E. Fajardo
 * Galileo Bionic RoboHand
 * Touch Screen Feedback
 * EMG Sampler
 * CMSIS-DSP Application
 * Prostheses Controller
 * main.c
 */
 
 //CMSIS-DSP Directives
#define ARM_MATH_CM4
#if !defined  (__FPU_PRESENT) 
  #define __FPU_PRESENT             1       
#endif 

#define TEST_LENGTH_SAMPLES  500

//#include "TM4C129ENCPDT.h"                    // Device header
#include <stdint.h>
#include <stdbool.h>
//#include <stdlib.h>
//#include <string.h>
//#include <math.h>
#include "inc/hw_ints.h"
#include "inc/hw_memmap.h"
#include "inc/hw_types.h"
#include "inc/hw_gpio.h"
#include "driverlib/fpu.h"
#include "driverlib/sysctl.h"
#include "driverlib/interrupt.h"
#include "driverlib/rom.h"
#include "driverlib/gpio.h"
#include "driverlib/adc.h"
#include "driverlib/timer.h"
#include "driverlib/uart.h"
#include "driverlib/pwm.h"
#include "driverlib/pin_map.h"
#include "grlib/grlib.h"
#include "grlib/widget.h"
#include "grlib/canvas.h"
#include "grlib/pushbutton.h"
#include "drivers/Kentec320x240x16_ssd2119_8bit.h"
#include "drivers/touch.h"
#include "utils/ustdlib.h"
#include "arm_math.h"  
#include "AI/NeuralNetworkFunction.h"

uint8_t estado = 0;
uint8_t command = 'x';
uint8_t recognized = 'a';
uint8_t sstate = 0;
//COCO FLAG
uint8_t COCO = 0;
uint8_t READY = 0;
uint16_t count = 0;
//Board Clock
uint32_t ui32SysClkFreq;
//ADC values [E2,E1,Ref]
uint32_t ulValue[3];
//Electrodes Variables
float32_t E1_Voltage;
float32_t E2_Voltage;
//25 milliseconds buffer
float32_t samples25ms[100];
float32_t samples25ms2[100];
//rectified signal 
float32_t abs_output25[100];
float32_t abs_output252[100];
//filtered signal
float32_t samples25fi[128]; //53//78//128
float32_t samples25fi2[128]; //53//78//128

float32_t samples[TEST_LENGTH_SAMPLES];
float32_t samples2[TEST_LENGTH_SAMPLES];
float32_t samples_minus_one[TEST_LENGTH_SAMPLES];
float32_t samples_plus_one[TEST_LENGTH_SAMPLES];
float32_t abs_outputF32[TEST_LENGTH_SAMPLES];
float32_t difm_outputF32[TEST_LENGTH_SAMPLES];
float32_t difp_outputF32[TEST_LENGTH_SAMPLES];
float32_t abs_difm_outputF32[TEST_LENGTH_SAMPLES];
float32_t abs_difp_outputF32[TEST_LENGTH_SAMPLES];
float32_t ones[TEST_LENGTH_SAMPLES];
//float32_t samplesf[TEST_LENGTH_SAMPLES+100-1];

//mean os the filtered signal
float32_t mean=0;
float32_t mean2=0;
float32_t rms=0;
float32_t rms2=0;
//action options
uint8_t action=0;
//string through serial port
char String[16];
static char text1[16];
static char text2[16];
static char text3[16];
static char text4[16];
static char text5[16];
static char text6[16];
//features
typedef struct feat{
  float32_t iemg;
	float32_t wl;
  float32_t var;
  int32_t wamp;
  int32_t zc;
  int32_t ssc;	
} features;

features feat1 = {0.0f,0.0f,0.0f,0,0,0};
features feat2 = {0.0f,0.0f,0.0f,0,0,0};

float y[5];																																				//neural network output
float x[12] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};			//neural network input
float32_t max = 0.0f;																															//max accuracy
uint32_t clas = 0;																																//class

//FIR impulse response
float32_t h[29] = {
	 0.005122283331907f,0.005887141527430f,0.008108906860497f,0.011692906924201f,0.016472293441426f,0.022216298876727f,0.028641952887652f,0.035428649080362f,
	 0.042234794064957f,0.048715653723188f,0.054541442583407f,0.059414685262420f,0.063085915202651f,0.065366863558607f,0.0661404253491338f,0.065366863558607f,
	 0.063085915202651f,0.059414685262420f,0.054541442583407f,0.048715653723188f,0.042234794064957f,0.035428649080362f,0.028641952887652f,0.022216298876727f,
	 0.016472293441426f,0.011692906924201f,0.008108906860497f,0.005887141527430f,0.005122283331907f
};
/*
float32_t h2[101] = {
	0.000000000000000f,0.000308983000000f,0.000527514000000f,0.000567861000000f,0.000384231000000f,0.000000000000000f,-0.000477615000000f,-0.000871582000000f,-0.000986912000000f,-0.000691961000000f,0.000000000000000f,
	0.000890367000000f,0.001630827000000f,0.001842119000000f,0.001282793000000f,0.000000000000000f,-0.001615469000000f,-0.002921311000000f,-0.003256229000000f,-0.002237471000000f,0.000000000000000f,
	0.002746291000000f,0.004907256000000f,0.005409080000000f,0.003678635000000f,0.000000000000000f,-0.004435600000000f,-0.007867728000000f,-0.008618170000000f,-0.005831261000000f,0.000000000000000f,
	0.006986174000000f,0.012377976000000f,0.013564684000000f,0.009198311000000f,0.000000000000000f,-0.011137027000000f,-0.019911115000000f,-0.022085957000000f,-0.015215872000000f,0.000000000000000f,
	0.019303167000000f,0.035694073000000f,0.041360697000000f,0.030180640000000f,0.000000000000000f,-0.046106072000000f,-0.100104485000000f,-0.150841107000000f,-0.186958764000000f,0.800131546000000f,
	-0.186958764000000f,-0.150841107000000f,-0.100104485000000f,-0.046106072000000f,0.000000000000000f,0.030180640000000f,0.041360697000000f,0.035694073000000f,0.019303167000000f,0.000000000000000f,
	-0.015215872000000f,-0.022085957000000f,-0.019911115000000f,-0.011137027000000f,0.000000000000000f,0.009198311000000f,0.013564684000000f,0.012377976000000f,0.006986174000000f,0.000000000000000f,
	-0.005831261000000f,-0.008618170000000f,-0.007867728000000f,-0.004435600000000f,0.000000000000000f,0.003678635000000f,0.005409080000000f,0.004907256000000f,0.002746291000000f,0.000000000000000f,
	-0.002237471000000f,-0.003256229000000f,-0.002921311000000f,-0.001615469000000f,0.000000000000000f,0.001282793000000f,0.001842119000000f,0.001630827000000f,0.000890367000000f,0.000000000000000f,
	-0.000691961000000f,-0.000986912000000f,-0.000871582000000f,-0.000477615000000f,0.000000000000000f,0.000384231000000f,0.000567861000000f,0.000527514000000f,0.000308983000000f,0.000000000000000f
};*/

//initial positions for the fingers
uint32_t thumbp = 1125;  uint32_t ttemp = 1125; 
uint32_t indexp = 1125;  uint32_t itemp = 1125; 
uint32_t middlep = 4500; uint32_t mtemp = 4500; 
uint32_t ringp = 4500;   uint32_t rtemp = 4500; 
uint32_t littlep = 1125; uint32_t ltemp = 1125; 

//canvas for the background screen
extern tCanvasWidget g_sMainMenu;
extern tCanvasWidget g_sMachineLearning;
extern tCanvasWidget g_sSpeechRecognized;
//push buttons for the screen
extern tPushButtonWidget g_sPushBtn;
extern tPushButtonWidget g_sPushBtn2;
extern tPushButtonWidget g_sPushBtn3;
extern tPushButtonWidget g_sPushBtn4;
extern tPushButtonWidget g_sPushBtn5;
extern tPushButtonWidget g_sPushBtn6;
extern tPushButtonWidget g_sPushBtn7;
extern tPushButtonWidget g_sPushBtn8;
extern tPushButtonWidget g_sPushBtn9;
extern tPushButtonWidget g_sPushBtn10;
extern tPushButtonWidget g_sPushBtn11;
extern tPushButtonWidget g_sPushBtn12;
extern tPushButtonWidget g_sPushBtn13;
extern tPushButtonWidget g_sPushBtn14;
extern tPushButtonWidget g_sPushBtn15;
extern tPushButtonWidget g_sPushBtn16;
extern tPushButtonWidget g_sPushBtn17;
tContext g_sContext;

//Event declaration for the screen
void OnButtonPress(tWidget *pWidget);
void OnButtonPress2(tWidget *pWidget);
void OnButtonPress3(tWidget *pWidget);
void OnButtonPress4(tWidget *pWidget);
void OnButtonPress5(tWidget *pWidget);
void OnButtonPress6(tWidget *pWidget);
void OnButtonPress7(tWidget *pWidget);
void OnButtonPress8(tWidget *pWidget);
void OnButtonPress9(tWidget *pWidget);
void OnButtonPress10(tWidget *pWidget);
void OnButtonPress11(tWidget *pWidget);
void OnButtonPress12(tWidget *pWidget);
void OnButtonPress13(tWidget *pWidget);
void OnButtonPress14(tWidget *pWidget);
void OnButtonPress15(tWidget *pWidget);
void OnButtonPress16(tWidget *pWidget);
void OnButtonPress17(tWidget *pWidget);
//Initialization declaration
void GPIO_Init(void);
void PWMs_Init(void);
void Timers_Init(void);
void ADC_Init(void);
void UART_init(void);
//Utilities
void reverse(char s[]);
void itoa(uint32_t n, char s[]);
void UART_OutString(char buffer[]);
//To positioning the fingers
void FingerPositionSet(uint32_t thumb, uint32_t index, uint32_t middle, uint32_t ring, uint32_t little);
uint32_t FingerVerification(uint32_t);
//AI
features GetFeatures(float32_t samples[TEST_LENGTH_SAMPLES]);
void Execute(uint32_t clas);

//Canvas(name, parent, next, child, 
// 			display, lx, ly, lwidth, lheight
//			Style, FillColor, OutlineColor, TextColor
//			Font, Text, Image, OnPaint)

//Root Heading Text
Canvas(g_sHeading, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 0, 25, 320, 23,
		(CANVAS_STYLE_FILL | CANVAS_STYLE_OUTLINE | CANVAS_STYLE_TEXT),
		0x003366, 0x003366, ClrWhite, g_psFontCmss24b, "Galileo Bionic Hand", 0, 0
		);

//Root Canvas
Canvas(g_sMainMenu, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 0, 0, 320, 240,
		CANVAS_STYLE_FILL, 0x003366, 0, 0, 0, 0, 0, 0
		);

//Machine Learning Heading Text
Canvas(g_sMachineLearningHeading, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 0, 25, 320, 23,
		(CANVAS_STYLE_FILL | CANVAS_STYLE_OUTLINE | CANVAS_STYLE_TEXT),
		0x003366, 0x003366, ClrWhite, g_psFontCmss24b, "Machine Learning", 0, 0
		);
//Machine Learning Text		
Canvas(g_sMachineLearningText, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 100, 10, 200, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_LEFT), 0, 0, ClrYellow, g_psFontCmss12, "Close", 0, 0
		);
Canvas(g_sMachineLearningText2, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 100, 35, 200, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_LEFT), 0, 0, ClrYellow, g_psFontCmss12, "Open", 0, 0
		);
Canvas(g_sMachineLearningText3, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 100, 60, 200, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_LEFT), 0, 0, ClrYellow, g_psFontCmss12, "Flexion ", 0, 0
		);
Canvas(g_sMachineLearningText4, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 100, 85, 200, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_LEFT), 0, 0, ClrYellow, g_psFontCmss12, "Extension", 0, 0
		);
Canvas(g_sMachineLearningText9, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 100, 110, 200, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_LEFT), 0, 0, ClrYellow, g_psFontCmss12, "Peace", 0, 0
		);	
//we can set the text with text1
Canvas(g_sMachineLearningText5, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 150, 10, 80, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_RIGHT), 0, 0, ClrWhite, g_psFontCmss12, text1, 0, 0
		);
Canvas(g_sMachineLearningText6, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 150, 35, 80, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_RIGHT), 0, 0, ClrWhite, g_psFontCmss12, text2, 0, 0
		);
Canvas(g_sMachineLearningText7, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 150, 60, 80, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_RIGHT), 0, 0, ClrWhite, g_psFontCmss12, text3, 0, 0
		);
Canvas(g_sMachineLearningText8, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 150, 85, 80, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_RIGHT), 0, 0, ClrWhite, g_psFontCmss12, text4, 0, 0
		);
Canvas(g_sMachineLearningText10, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 150, 110, 80, 100,
		(CANVAS_STYLE_TEXT|CANVAS_STYLE_TEXT_RIGHT), 0, 0, ClrWhite, g_psFontCmss12, text5, 0, 0
		);		

//Machine Learning Canvas
Canvas(g_sMachineLearning, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 0, 0, 320, 240,
		CANVAS_STYLE_FILL, 0x003366, 0, 0, 0, 0, 0, 0
		);

//Speech Recognized Heading Text
Canvas(g_sSpeechRecognizedHeading, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 0, 25, 320, 23,
		(CANVAS_STYLE_FILL | CANVAS_STYLE_OUTLINE | CANVAS_STYLE_TEXT),
		0x003366, 0x003366, ClrWhite, g_psFontCmss24b, "Speech Recognition", 0, 0
		);
//Speech Recognized Text		
Canvas(g_sSpeechRecognizedText, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 60, 30, 200, 100,
		CANVAS_STYLE_TEXT, 0, 0, ClrYellow, g_psFontCmss22, text6, 0, 0
		);
//Speech Recognized Canvas
Canvas(g_sSpeechRecognized, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 0, 0, 320, 240,
		CANVAS_STYLE_FILL, 0x003366, 0, 0, 0, 0, 0, 0
		);
		
//On Line Learning Heading Text
Canvas(g_sOnLineHeading, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 0, 25, 320, 23,
		(CANVAS_STYLE_FILL | CANVAS_STYLE_OUTLINE | CANVAS_STYLE_TEXT),
		0x003366, 0x003366, ClrWhite, g_psFontCmss24b, "On Line Learning", 0, 0
		);

//On Line Learning Canvas
Canvas(g_sOnLine, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 0, 0, 320, 240,
		CANVAS_STYLE_FILL, 0x003366, 0, 0, 0, 0, 0, 0
		);

//RectangularButton(name, Parent, Next, Child,
//		Display, lX, lY, lWidth, lHeight,
//		Style, FillColor, PressFillColor, OutlineColor, TextColor,
//		Font, Text, Image, PressImage, AutoRepeatDelay, AutoRepeatRate, OnButtonPress);

//Root Canvas Buttons
RectangularButton(g_sPushBtn, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 30, 75, 120, 40,
		(PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
		PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
		g_psFontCmss22b, "Close", 0, 0, 0, 0, OnButtonPress
		);
		
RectangularButton(g_sPushBtn2, 0, 0, 0,
    &g_sKentec320x240x16_SSD2119, 30, 125, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
    g_psFontCmss22b, "Point", 0, 0, 0, 0, OnButtonPress2
		);

RectangularButton(g_sPushBtn3, 0, 0, 0,									
    &g_sKentec320x240x16_SSD2119, 30, 175, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
    g_psFontCmss22b, "Peace", 0, 0, 0, 0, OnButtonPress3
		);

RectangularButton(g_sPushBtn4, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 180, 75, 120, 40,
		(PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
		PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
		g_psFontCmss22b, "Pinch", 0, 0, 0, 0, OnButtonPress4
		);
//Machine Learning Menu		
RectangularButton(g_sPushBtn5, 0, 0, 0,
    &g_sKentec320x240x16_SSD2119, 180, 125, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x00CCCC, ClrGray, ClrWhite,
    g_psFontCmss22b, "ANN", 0, 0, 0, 0, OnButtonPress5
		);
//Speech Menu - the last one doesn't have a brother
RectangularButton(g_sPushBtn6, 0, 0, 0,								
    &g_sKentec320x240x16_SSD2119, 180, 175, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x00CCCC, ClrGray, ClrWhite,
    g_psFontCmss22b, "Speech", 0, 0, 0, 0, OnButtonPress6
		);
//Back Main Menu  
RectangularButton(g_sPushBtn7, 0, 0, 0,								
    &g_sKentec320x240x16_SSD2119, 30, 175, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x00CCCC, ClrGray, ClrWhite,
    g_psFontCmss22b, "Back", 0, 0, 0, 0, OnButtonPress7
		);
//On Line Learning Menu
RectangularButton(g_sPushBtn8, 0, 0, 0,								
    &g_sKentec320x240x16_SSD2119, 180, 175, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x00CCCC, ClrGray, ClrWhite,
    g_psFontCmss22b, "Learn", 0, 0, 0, 0, OnButtonPress8
		);

RectangularButton(g_sPushBtn9, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 30, 75, 120, 40,
		(PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
		PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
		g_psFontCmss22b, "Open", 0, 0, 0, 0, OnButtonPress9
		);
		
RectangularButton(g_sPushBtn10, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 30, 125, 120, 40,
		(PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
		PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
		g_psFontCmss22b, "Close", 0, 0, 0, 0, OnButtonPress10
		);

RectangularButton(g_sPushBtn11, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 180, 75, 120, 40,
		(PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
		PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
		g_psFontCmss22b, "Flexion", 0, 0, 0, 0, OnButtonPress11
		);

RectangularButton(g_sPushBtn12, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 180, 125, 120, 40,
		(PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
		PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
		g_psFontCmss22b, "Extension", 0, 0, 0, 0, OnButtonPress12
		);
		
RectangularButton(g_sPushBtn13, 0, 0, 0,
		&g_sKentec320x240x16_SSD2119, 180, 175, 120, 40,
		(PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
		PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
		g_psFontCmss22b, "Save", 0, 0, 0, 0, OnButtonPress13
		);
		
RectangularButton(g_sPushBtn14, 0, 0, 0,								
    &g_sKentec320x240x16_SSD2119, 30, 175, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x00CCCC, ClrGray, ClrWhite,
    g_psFontCmss22b, "Back", 0, 0, 0, 0, OnButtonPress14
		);

RectangularButton(g_sPushBtn15, 0, 0, 0,
    &g_sKentec320x240x16_SSD2119, 30, 125, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x009999, ClrGray, ClrWhite,
    g_psFontCmss22b, "English", 0, 0, 0, 0, OnButtonPress15
		);

RectangularButton(g_sPushBtn16, 0, 0, 0,
    &g_sKentec320x240x16_SSD2119, 180, 125, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x00CCCC, ClrGray, ClrWhite,
    g_psFontCmss22b, "Spanish", 0, 0, 0, 0, OnButtonPress16
		);
 
 RectangularButton(g_sPushBtn17, 0, 0, 0,								
    &g_sKentec320x240x16_SSD2119, 180, 175, 120, 40,
    (PB_STYLE_OUTLINE | PB_STYLE_TEXT_OPAQUE | PB_STYLE_TEXT |
    PB_STYLE_FILL), 0x00CCCC, 0x00CCCC, ClrGray, ClrWhite,
    g_psFontCmss22b, "Japanese", 0, 0, 0, 0, OnButtonPress17
		);

int32_t main(void) {
	//Floating Point Unit Init
	FPULazyStackingEnable();
	FPUEnable();
	
	//Clock Frequency at 120 MHz using PLL
	ui32SysClkFreq = SysCtlClockFreqSet((SYSCTL_XTAL_25MHZ | SYSCTL_OSC_MAIN | SYSCTL_USE_PLL | SYSCTL_CFG_VCO_480), 120000000);
	
	//Kentec Display & Touch Screen Init
	Kentec320x240x16_SSD2119Init(ui32SysClkFreq);
	TouchScreenInit(ui32SysClkFreq);
	
	TouchScreenCallbackSet(WidgetPointerMessage);
	
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMainMenu);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sHeading);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn2);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn3);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn4);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn5);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn6);
	WidgetPaint(WIDGET_ROOT);
	
	//Master enable for all the interrupts
	IntMasterEnable();																						
	
	//Peripheral Initializations
	GPIO_Init();
	PWMs_Init();
	Timers_Init();
	ADC_Init();
	UART_init();
	
	arm_fill_f32(1.0f, ones, TEST_LENGTH_SAMPLES);
	
	UARTCharPut(UART0_BASE, 'l');
	UARTCharPut(UART0_BASE, 'E');
	UARTCharPut(UART0_BASE, 'o');
	UARTCharPut(UART0_BASE, 'D');
	COCO = 0;
		
	while(1) {
		WidgetMessageQueueProcess();
		if(COCO){
			//mean for threshold
			arm_abs_f32(samples25ms, abs_output25, 100);											//rectifier
			arm_conv_f32(abs_output25,100,h,29,samples25fi);									//low pass filter
			arm_abs_f32(samples25ms2, abs_output252, 100);										//rectifier
			arm_conv_f32(abs_output252,100,h,29,samples25fi2);								//low pass filter
			//arm_mean_f32(samples25fi, 78, &mean);														//mean
			arm_rms_f32(samples25fi, 128, &rms);															//rms
			arm_rms_f32(samples25fi2, 128, &rms2);														//128													
			switch(action){																											
				case 1: {
					switch (estado){
						case 0: {
							if (rms>0.145f){
								FingerPositionSet(1125,1125,1125,4500,4500);										//close
								//UARTCharPut(UART0_BASE, '2');
								//UARTCharPut(UART0_BASE, '\r');
								GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x03);
								estado = 1;
							}
						} break;
						case 1: {
							if (rms2>0.145f){
								FingerPositionSet(4500,4500,4500,1125,1125);										//open
								//UARTCharPut(UART0_BASE, '0');
								//UARTCharPut(UART0_BASE, '\r');
								GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x00);
								estado = 0;
							}								
						} break;
					}
				} break; 				
				case 2: {																													
					switch (estado){
						case 0: {
							if (rms>0.145f){
								FingerPositionSet(1125,4500,1125,4500,4500); 										//point
								//UARTCharPut(UART0_BASE, '1');
								//UARTCharPut(UART0_BASE, '\r');
								GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x03);
								estado = 1;
							}
						} break;
						case 1: {
							if (rms2>0.145f){
								FingerPositionSet(4500,4500,4500,1125,1125);										//open
								//UARTCharPut(UART0_BASE, '0');
								//UARTCharPut(UART0_BASE, '\r');
								GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x00);
								estado = 0;
							}								
						} break;
					}
				} break;
						
				case 3: {																														
					switch (estado){
						case 0: {
							if (rms>0.145f){
								FingerPositionSet(1125,4500,4500,4500,4500); 										//peace
								//UARTCharPut(UART0_BASE, '3');
								//UARTCharPut(UART0_BASE, '\r');
								GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x03);
								estado = 1;
							}
						} break;
						case 1: {
							if (rms2>0.145f){
								FingerPositionSet(4500,4500,4500,1125,1125);										//open
								GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x00);
								//UARTCharPut(UART0_BASE, '0');
								//UARTCharPut(UART0_BASE, '\r');
								estado = 0;
							}								
						} break;
					}
				} break;					
				case 4: {																													
					switch (estado){
						case 0: {
							if (rms>0.125f){
								FingerPositionSet(1125,1125,4500,1125,4500);										//pinch
								//UARTCharPut(UART0_BASE, '4');
								//UARTCharPut(UART0_BASE, '\r');
								GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x03);
								estado = 1;
							}
						} break;
						case 1: {
							if (rms2>0.145f){
								FingerPositionSet(4500,4500,4500,1125,1125);										//open
								GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x00);
								//UARTCharPut(UART0_BASE, '0');
								//UARTCharPut(UART0_BASE, '\r');
								estado = 0;
							}								
						} break;
					}
				} break; 				
				case 5: {																													//ann
					if ((rms>0.135f)||(rms2>0.135f)){															//0.0875f
						GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x03);
						if (!READY){
							  count++;
							  for(int j=TEST_LENGTH_SAMPLES-1;j>0;j--){
									samples[j]=samples[j-1];
									samples2[j]=samples2[j-1];
								}
								samples[0]=E1_Voltage;
								samples2[0]=E2_Voltage;
								READY = (count<400)?0:1;
						} else {
							count = 0;
							READY = 0;
							//arm_conv_f32(samples,500,h,101,samplesf);								//high pass filter
							feat1 = GetFeatures(samples);
							feat2 = GetFeatures(samples2);	
							x[0] = feat1.iemg; 
							x[1] = feat1.wl; 
							x[2] = feat1.var;
							x[3] = (float32_t)feat1.wamp; 
							x[4]= (float32_t)feat1.zc; 
							x[5]= (float32_t)feat1.ssc;
							x[6] = feat2.iemg; 
							x[7] = feat2.wl; 
							x[8] = feat2.var;
							x[9] = (float32_t)feat2.wamp; 
							x[10] = (float32_t)feat2.zc; 
							x[11] = (float32_t)feat2.ssc;
							NeuralNetworkFunction(x,y);
							arm_max_f32(y,5,&max,&clas);
							Execute(clas);
							usprintf(text1, "%3d", (int)(y[0]*100));
							usprintf(text2, "%3d", (int)(y[1]*100));
							usprintf(text3, "%3d", (int)(y[2]*100));
							usprintf(text4, "%3d", (int)(y[3]*100));
							usprintf(text5, "%3d", (int)(y[4]*100));
							WidgetPaint(WIDGET_ROOT);
						}
					} else {
						GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x00);
					}
					//Execute(clas);
				}	break;					
				case 6: {																												//speech
					if ((rms>0.145f)||(rms2>0.145f)){
						GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x03);
						switch(recognized){
							//(rotation,pinky,index,middle,thumb)
							case 0: FingerPositionSet(1125,1125,1125,4500,4500); break; 	//close
							case 1: FingerPositionSet(1125,4500,1125,4500,4500); break;		//point
							case 2: FingerPositionSet(1125,4500,4500,4500,4500); break;		//peace
							case 3: FingerPositionSet(1125,1125,4500,1125,4500); break; 	//three
							case 4: FingerPositionSet(1125,4500,4500,1125,4500); break;		//four
							case 5: FingerPositionSet(4500,4500,4500,1125,1125); break;		//open
							default:  FingerPositionSet(1125,1125,1125,4500,4500); break; 	//close
						}
					} else {
						FingerPositionSet(1125,4500,4500,1125,1125);										//open
						GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x00);
						//UARTCharPut(UART0_BASE, '0');
						//UARTCharPut(UART0_BASE, '\r');
					}	
				} break;
				default:{
					if ((rms>0.145f)||(rms2>0.145f)){																	//close
						FingerPositionSet(1125,1125,1125,4500,4500);
						GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x03);
					} else {
						FingerPositionSet(4500,4500,4500,1125,1125);										//open
						GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x00);
					}
				}
			}
			COCO = 0;
		}
	}
}

//Touch Screen Events
void OnButtonPress(tWidget *pWidget) { action = 1; }	//close
void OnButtonPress2(tWidget *pWidget){ action = 2; } 	//point
void OnButtonPress3(tWidget *pWidget){ action = 3; } 	//rock
void OnButtonPress4(tWidget *pWidget){ action = 4; }	//peace
void OnButtonPress5(tWidget *pWidget){ 								//ANN
	action = 5;
	WidgetRemove((tWidget *)&g_sMainMenu);
	WidgetRemove((tWidget *)&g_sHeading);
	WidgetRemove((tWidget *)&g_sPushBtn);
	WidgetRemove((tWidget *)&g_sPushBtn2);
	WidgetRemove((tWidget *)&g_sPushBtn3);
	WidgetRemove((tWidget *)&g_sPushBtn4);
	WidgetRemove((tWidget *)&g_sPushBtn5);
	WidgetRemove((tWidget *)&g_sPushBtn6);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearning);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningHeading);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText2);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText3);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText4);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText5);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText6);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText7);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText8);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText9);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText10);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn7);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn8);
	WidgetPaint(WIDGET_ROOT);
}	
void OnButtonPress6(tWidget *pWidget){ 								//Speech
	action = 6;
	WidgetRemove((tWidget *)&g_sMainMenu);
	WidgetRemove((tWidget *)&g_sHeading);
	WidgetRemove((tWidget *)&g_sPushBtn);
	WidgetRemove((tWidget *)&g_sPushBtn2);
	WidgetRemove((tWidget *)&g_sPushBtn3);
	WidgetRemove((tWidget *)&g_sPushBtn4);
	WidgetRemove((tWidget *)&g_sPushBtn5);
	WidgetRemove((tWidget *)&g_sPushBtn6);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sSpeechRecognized);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sSpeechRecognizedHeading);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sSpeechRecognizedText);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn7);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn15);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn16);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn17);
	WidgetPaint(WIDGET_ROOT);
}	

void OnButtonPress7(tWidget *pWidget){ 								//Back
  action = 1;
	WidgetRemove((tWidget *)&g_sMachineLearning);
	WidgetRemove((tWidget *)&g_sMachineLearningHeading);
	WidgetRemove((tWidget *)&g_sMachineLearningText);
	WidgetRemove((tWidget *)&g_sMachineLearningText2);
	WidgetRemove((tWidget *)&g_sMachineLearningText3);
	WidgetRemove((tWidget *)&g_sMachineLearningText4);
	WidgetRemove((tWidget *)&g_sMachineLearningText5);
	WidgetRemove((tWidget *)&g_sMachineLearningText6);
	WidgetRemove((tWidget *)&g_sMachineLearningText7);
	WidgetRemove((tWidget *)&g_sMachineLearningText8);
	WidgetRemove((tWidget *)&g_sMachineLearningText9);
	WidgetRemove((tWidget *)&g_sMachineLearningText10);
	WidgetRemove((tWidget *)&g_sSpeechRecognized);
	WidgetRemove((tWidget *)&g_sSpeechRecognizedHeading);
	WidgetRemove((tWidget *)&g_sSpeechRecognizedText);
	WidgetRemove((tWidget *)&g_sPushBtn7);
	WidgetRemove((tWidget *)&g_sPushBtn8);
	WidgetRemove((tWidget *)&g_sPushBtn15);
	WidgetRemove((tWidget *)&g_sPushBtn16);
	WidgetRemove((tWidget *)&g_sPushBtn17);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMainMenu);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sHeading);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn2);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn3);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn4);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn5);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn6);
	WidgetPaint(WIDGET_ROOT);
}

void OnButtonPress8(tWidget *pWidget){ 								//Learn
  action = 8;
	WidgetRemove((tWidget *)&g_sMachineLearning);
	WidgetRemove((tWidget *)&g_sMachineLearningHeading);
	WidgetRemove((tWidget *)&g_sMachineLearningText);
	WidgetRemove((tWidget *)&g_sMachineLearningText2);
	WidgetRemove((tWidget *)&g_sMachineLearningText3);
	WidgetRemove((tWidget *)&g_sMachineLearningText4);
	WidgetRemove((tWidget *)&g_sMachineLearningText5);
	WidgetRemove((tWidget *)&g_sMachineLearningText6);
	WidgetRemove((tWidget *)&g_sMachineLearningText7);
	WidgetRemove((tWidget *)&g_sMachineLearningText8);
	WidgetRemove((tWidget *)&g_sMachineLearningText9);
	WidgetRemove((tWidget *)&g_sMachineLearningText10);
	WidgetRemove((tWidget *)&g_sPushBtn7);
	WidgetRemove((tWidget *)&g_sPushBtn8);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sOnLine);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sOnLineHeading);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn9);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn10);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn11);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn12);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn13);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn14);
	WidgetPaint(WIDGET_ROOT);
}

void OnButtonPress9(tWidget *pWidget) { action = 9; }		//open
void OnButtonPress10(tWidget *pWidget){ action = 10; } 	//close
void OnButtonPress11(tWidget *pWidget){ action = 11; } 	//flexion
void OnButtonPress12(tWidget *pWidget){ action = 12; }	//extension
void OnButtonPress13(tWidget *pWidget){ action = 13; }	//point

void OnButtonPress14(tWidget *pWidget){ 								//back
	action = 5; 
	WidgetRemove((tWidget *)&g_sOnLine);
	WidgetRemove((tWidget *)&g_sOnLineHeading);
	WidgetRemove((tWidget *)&g_sPushBtn9);
	WidgetRemove((tWidget *)&g_sPushBtn10);
	WidgetRemove((tWidget *)&g_sPushBtn11);
	WidgetRemove((tWidget *)&g_sPushBtn12);
	WidgetRemove((tWidget *)&g_sPushBtn13);
	WidgetRemove((tWidget *)&g_sPushBtn14);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearning);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningHeading);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText2);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText3);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText4);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText5);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText6);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText7);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText8);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText9);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sMachineLearningText10);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn7);
	WidgetAdd(WIDGET_ROOT,(tWidget *)&g_sPushBtn8);
	WidgetPaint(WIDGET_ROOT);
}	

void OnButtonPress15(tWidget *pWidget){ //english
	action = 6;
	UARTCharPut(UART0_BASE, 'l');
	UARTCharPut(UART0_BASE, 'A');
}	
void OnButtonPress16(tWidget *pWidget){	//spanish 
	action = 6;
	UARTCharPut(UART0_BASE, 'l');
	UARTCharPut(UART0_BASE, 'E');
}	
void OnButtonPress17(tWidget *pWidget){ //japanese
	action = 6;
	UARTCharPut(UART0_BASE, 'l');
	UARTCharPut(UART0_BASE, 'C');	
}	

//Finger positioning
void FingerPositionSet(uint32_t thumb, uint32_t index, uint32_t middle, uint32_t ring, uint32_t little){
	thumbp=thumb; indexp=index; middlep=middle; ringp=ring; littlep=little;
}

uint32_t FingerVerification(uint32_t position){
	uint32_t temp;
	uint32_t m = __get_PRIMASK();
	__disable_irq();
	if(temp>position) {temp=position-15;}									//4500-1125->3375/15=225
	if(temp<position) {temp=position+15;}									
	if(temp==position) {temp=position; }
	__set_PRIMASK(m);
	return temp;
}

//get features
features GetFeatures(float32_t samples[TEST_LENGTH_SAMPLES]){
	
	volatile features feat;
	float32_t iemg, var, wl;
  int32_t wamp, zc, ssc, sgn;
	
	for(int k=0;k<TEST_LENGTH_SAMPLES-1;k++) samples_minus_one[k]=samples[k+1];
	samples_minus_one[TEST_LENGTH_SAMPLES-1]=0.0f;
			
	for(int j=TEST_LENGTH_SAMPLES-1;j>0;j--) samples_plus_one[j]=samples[j-1];
	samples_plus_one[0]=0.0f;
			
	/*IEMG*/
	arm_abs_f32(samples, abs_outputF32, TEST_LENGTH_SAMPLES);	
	arm_dot_prod_f32(abs_outputF32, ones, TEST_LENGTH_SAMPLES, &iemg);
			
	/*waveform length*/	
	wl=0.0f;
	arm_sub_f32(samples, samples_minus_one, difm_outputF32, TEST_LENGTH_SAMPLES);
	arm_abs_f32(difm_outputF32, abs_difm_outputF32, TEST_LENGTH_SAMPLES);	
	arm_dot_prod_f32(abs_difm_outputF32, ones, TEST_LENGTH_SAMPLES, &wl);
			
	/*variance*/
	arm_var_f32(samples, TEST_LENGTH_SAMPLES, &var);
	
	/*wamp*/
	wamp=0;
	arm_sub_f32(samples, samples_plus_one, difp_outputF32, TEST_LENGTH_SAMPLES);
	arm_abs_f32(difp_outputF32, abs_difp_outputF32, TEST_LENGTH_SAMPLES);
	for(int m=0;m<TEST_LENGTH_SAMPLES;m++) if(abs_difp_outputF32[m]>=0.06f) wamp++;
			
	/*zc*/
	zc=0; 
	for(int n=0;n<TEST_LENGTH_SAMPLES-1;n++) {
		sgn = ((-samples[n]*samples[n+1])>0)?1:0; 
		if ((sgn>0)&&(abs_difp_outputF32[n]>0.06f)) zc++;
	}
			
	/*ssc*/
	ssc=0;
	for(int n=0;n<TEST_LENGTH_SAMPLES;n++) if (((samples[n]-samples_minus_one[n])*(samples[n]-samples_plus_one[n]))>0.06f) ssc++;
			
	feat.iemg = iemg;
	feat.wl = wl;
	feat.var = var;
  feat.wamp = wamp;
  feat.zc = zc;
  feat.ssc = ssc;
	
	return feat;
}

void Execute(uint32_t clas){
	switch (clas){
		case 0:{
			//UART_OutString("2\r"); 
			//UART_OutString("close!!!"); UARTCharPut(UART0_BASE, '\r');
			//(rotation,pinky,index,middle,thumb)
			FingerPositionSet(1125,1125,1125,4500,4500);										//close
		} break;
		case 1:{
			//UART_OutString("0\r");
			//UART_OutString("open!!!"); UARTCharPut(UART0_BASE, '\r');
			//(rotation,pinky,index,middle,thumb)
			FingerPositionSet(4500,4500,4500,1125,1125);					//open
		} break;
		case 2:{
			//UART_OutString("1\r"); 
			//UART_OutString("point!!!"); UARTCharPut(UART0_BASE, '\r');
			//(rotation,pinky,index,middle,thumb)
			FingerPositionSet(1125,4500,1125,4500,4500); 					//point
		} break;
		case 3:{
			//UART_OutString("3\r");
			//UART_OutString("peace!!!"); UARTCharPut(UART0_BASE, '\r');
			//(rotation,pinky,index,middle,thumb)	rotacion
			FingerPositionSet(1125,1125,4500,1125,4500); 					//pinch
		} break;
		case 4:{
			//UART_OutString("4\r");
			//UART_OutString("peace!!!"); UARTCharPut(UART0_BASE, '\r');
			//(rotation,pinky,index,middle,thumb) cambiar rotacion
			FingerPositionSet(1125,1125,1125,4500,1125); 					//lateral
		} 
	}
}

//array reverse
void reverse(char s[]){
	int i, j; char c;
  for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
		c = s[i];
    s[i] = s[j];
    s[j] = c;	
	}
}

//int to string
void itoa(uint32_t n, char s[]){
	int i, sign;
  if ((sign = n) < 0)  											/* record sign */
		n = -n;          												/* make n positive */
    i = 0;
    do {      														  /* generate digits in reverse order */
			s[i++] = n % 10 + '0';   							/* get next digit */
    } while ((n /= 10) > 0);     						/* delete it */
			if (sign < 0)
				s[i++] = '-';
			s[i] = '\0';
			reverse(s);
}


void UART_OutString(char buffer[]){
	while(*buffer){
		//UARTCharPutNonBlocking(UART0_BASE, *buffer);
		UARTCharPut(UART0_BASE, *buffer);
		buffer++;
	}
}

void ADC1SS2_Handler(void){
}

//push button interrupt service button
void GPIO_PORTJ_Handler(void){
	if (GPIOIntStatus(GPIO_PORTJ_BASE, false) & GPIO_PIN_0) {	
		if (GPIOPinRead(GPIO_PORTJ_BASE, GPIO_PIN_0)){
			UARTCharPut(UART0_BASE, 'i');
			UARTCharPut(UART0_BASE, 'D');
			GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_0, 0x01);
			GPIOPinWrite(GPIO_PORTL_BASE, GPIO_PIN_0, 0x01);
			SysCtlDelay(500000);
			GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_0, 0x00);
			GPIOPinWrite(GPIO_PORTL_BASE, GPIO_PIN_0, 0x00);
		}
		GPIOIntClear(GPIO_PORTJ_BASE, GPIO_PIN_0);
	}
}

//thumb interrupt service routine - change velocity with timer
void Timer0A_Handler(void){
	// Clear the timer interrupt source
	TimerIntClear(TIMER0_BASE, TIMER_TIMA_TIMEOUT);
	ttemp = FingerVerification(thumbp);
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_6, ttemp);
	//PWMPulseWidthSet(PWM0_BASE, PWM_OUT_6, thumbp);
}
//index interrupt service routine - change velocity with timer
void Timer2A_Handler(void){
	// Clear the timer interrupt source
	TimerIntClear(TIMER2_BASE, TIMER_TIMA_TIMEOUT);
	itemp = FingerVerification(indexp);
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_4, itemp);
	//PWMPulseWidthSet(PWM0_BASE, PWM_OUT_4, indexp);
}
//middle interrupt service routine - change velocity with timer
void Timer3A_Handler(void){
	// Clear the timer interrupt source
	TimerIntClear(TIMER3_BASE, TIMER_TIMA_TIMEOUT);
	mtemp = FingerVerification(middlep);
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_2, mtemp);
	//PWMPulseWidthSet(PWM0_BASE, PWM_OUT_2, middlep);
}
//ring interrupt service routine - change velocity with timer
void Timer4A_Handler(void){
	// Clear the timer interrupt source
	TimerIntClear(TIMER4_BASE, TIMER_TIMA_TIMEOUT);
	rtemp = FingerVerification(ringp);
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_1, rtemp);
	//PWMPulseWidthSet(PWM0_BASE, PWM_OUT_1, ringp);
}
//little interrupt service routine - change velocity with timer
void Timer5A_Handler(void){
	// Clear the timer interrupt source
	TimerIntClear(TIMER5_BASE, TIMER_TIMA_TIMEOUT);
	ltemp = FingerVerification(littlep);
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_3, ltemp);
	//PWMPulseWidthSet(PWM0_BASE, PWM_OUT_3, littlep);
}

//ADC interrupt service routine - Sample Rate at 1 kHz
void Timer6A_Handler(void){
	TimerIntClear(TIMER6_BASE, TIMER_TIMA_TIMEOUT);
	ADCIntClear(ADC1_BASE, 2);
	ADCProcessorTrigger(ADC1_BASE, 2);
	while(!ADCIntStatus(ADC1_BASE, 2, false)){}											//Wait for the conversion to complete
	
	ADCSequenceDataGet(ADC1_BASE, 2, ulValue);											//Get Data
		
	E1_Voltage = ((int32_t)(ulValue[1]-ulValue[2])*3.3f)/4095.0f;		//Palmaris Longus
	E2_Voltage = ((int32_t)(ulValue[0]-ulValue[2])*3.3f)/4095.0f;		//Extonsorum Digitorum
	
	//uncomment send data 
	//itoa(ulValue[1],String);		
	//UART_OutString(String);	
	//UARTCharPut(UART0_BASE, '\r');
	
	uint32_t m = __get_PRIMASK();
	__disable_irq();		
	for(int j=100-1;j>0;j--) {
		samples25ms[j]=samples25ms[j-1];															//Fill Array
		samples25ms2[j]=samples25ms2[j-1];														//Fill Array
	}
	samples25ms[0]=E1_Voltage;
	samples25ms2[0]=E2_Voltage;
	//copy 100 samples on array
	for(int j=99;j>0;j--){ 
		samples[j]=samples25ms[j-1];
		samples2[j]=samples25ms2[j-1];
	}		
	__set_PRIMASK(m);
	COCO = 1;
		
}

void UARTIntHandler(void){
	uint32_t ui32Status;
	ui32Status = UARTIntStatus(UART0_BASE, true); //get interrupt status
	UARTIntClear(UART0_BASE, ui32Status); //clear the asserted interrupts
	while(UARTCharsAvail(UART0_BASE)) //loop while there are chars
		command = UARTCharGetNonBlocking(UART0_BASE); 
	switch (sstate){
		case 0:{
			if (command=='s')	{
				UARTCharPut(UART0_BASE,' ');
				sstate = 1;
			} else if (command=='t') {
				sstate = 0;
				if (action == 6) {
					usprintf(text6, "%s", "\"timeout\"");
					WidgetPaint(WIDGET_ROOT);
				}
			} else if (command=='e') {
				sstate = 0;
				if (action == 6) {
					usprintf(text6, "%s", "\"error\"");
					WidgetPaint(WIDGET_ROOT);
				}
			} else if (command=='o') {
				sstate = 0;
				if (action == 6) {
					usprintf(text6, "%s", "\"ok\"");
					WidgetPaint(WIDGET_ROOT);
				}
			} else {
				sstate = 0;
			}
		} break;
		case 1:{
			if (action == 6){
				recognized = command - 'A';
				switch(recognized){
					case 0: usprintf(text6, "%s", "\"zero\""); break;
					case 1: usprintf(text6, "%s", "\"one\""); break;
					case 2: usprintf(text6, "%s", "\"two\""); break;
					case 3: usprintf(text6, "%s", "\"three\""); break;
					case 4: usprintf(text6, "%s", "\"four\""); break;
					case 5: usprintf(text6, "%s", "\"five\""); break;
					case 6: usprintf(text6, "%s", "\"six\""); break;
					case 7: usprintf(text6, "%s", "\"seven\""); break;
					case 8: usprintf(text6, "%s", "\"eight\""); break;
					case 9: usprintf(text6, "%s", "\"nine\""); break;
					case 10: usprintf(text6, "%s", "\"ten\""); break;
				}
				WidgetPaint(WIDGET_ROOT);
			}	
			sstate = 0;
		}
	}
	
}

void GPIO_Init(void){
	//GPIO Init User - LEDs
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOJ);
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOL);
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPION);
	GPIOPinTypeGPIOOutput(GPIO_PORTF_BASE, GPIO_PIN_0);																				//for testing
	GPIOPinTypeGPIOInput(GPIO_PORTJ_BASE, GPIO_PIN_0);																				//for button
	GPIOPinTypeGPIOOutput(GPIO_PORTL_BASE, GPIO_PIN_0);																				//for beaglebone
	GPIOPinTypeGPIOOutput(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1);
	GPIOPadConfigSet(GPIO_PORTJ_BASE, GPIO_PIN_0, GPIO_STRENGTH_2MA, GPIO_PIN_TYPE_STD_WPU);	//pull up and drive strength
	GPIOPinWrite(GPIO_PORTF_BASE, GPIO_PIN_0, 0x00);
	GPIOPinWrite(GPIO_PORTL_BASE, GPIO_PIN_0, 0x00);
	GPIOPinWrite(GPIO_PORTN_BASE, GPIO_PIN_0|GPIO_PIN_1, 0x00);	
	GPIOIntTypeSet(GPIO_PORTJ_BASE, GPIO_PIN_0, GPIO_RISING_EDGE);														//Interrupt on falling edge
	GPIOIntEnable(GPIO_PORTJ_BASE, GPIO_PIN_0);																								//Port J Interrupt
	IntEnable(INT_GPIOJ);																																			//Enable the specific vector for Timer0A
}

void PWMs_Init(void){
	//PWM Init for Servo Motors
	//SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOF);
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOG);
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOK);
	SysCtlPeripheralEnable(SYSCTL_PERIPH_PWM0);
	//Clock division of 64
	PWMClockSet(PWM0_BASE,PWM_SYSCLK_DIV_64);
	//Configuring pins as PWM
	GPIOPinConfigure(GPIO_PF1_M0PWM1);
	GPIOPinConfigure(GPIO_PF2_M0PWM2);
	GPIOPinConfigure(GPIO_PF3_M0PWM3);
	GPIOPinConfigure(GPIO_PG0_M0PWM4);
	GPIOPinConfigure(GPIO_PK4_M0PWM6);
	
	GPIOPinTypePWM(GPIO_PORTF_BASE, GPIO_PIN_1);
	GPIOPinTypePWM(GPIO_PORTF_BASE, GPIO_PIN_2);
	GPIOPinTypePWM(GPIO_PORTF_BASE, GPIO_PIN_3);
	GPIOPinTypePWM(GPIO_PORTG_BASE, GPIO_PIN_0);
	GPIOPinTypePWM(GPIO_PORTK_BASE, GPIO_PIN_4);
	
	//PWM Generation
	PWMGenConfigure(PWM0_BASE, PWM_GEN_0, PWM_GEN_MODE_DOWN | PWM_GEN_MODE_NO_SYNC);
	PWMGenConfigure(PWM0_BASE, PWM_GEN_1, PWM_GEN_MODE_DOWN | PWM_GEN_MODE_NO_SYNC);
	PWMGenConfigure(PWM0_BASE, PWM_GEN_2, PWM_GEN_MODE_DOWN | PWM_GEN_MODE_NO_SYNC);
	PWMGenConfigure(PWM0_BASE, PWM_GEN_3, PWM_GEN_MODE_DOWN | PWM_GEN_MODE_NO_SYNC);
	//(120MHz/64)/50Hz - 20ms of period
	PWMGenPeriodSet(PWM0_BASE, PWM_GEN_0, 37500-1);		
	PWMGenPeriodSet(PWM0_BASE, PWM_GEN_1, 37500-1);		
	PWMGenPeriodSet(PWM0_BASE, PWM_GEN_2, 37500-1);											
	PWMGenPeriodSet(PWM0_BASE, PWM_GEN_3, 37500-1);											
	//Pulse width initialization
	/*
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_1, 1125);			//0.6 ms (1125) is close - 2.4 ms (4500) is open - ring
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_2, 4500);			//0.6 ms (1125) is close - 2.4 ms (4500) is open - middle
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_3, 1125);			//2.4 ms (1125) is close - 0.6 ms (4500) is open - little
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_4, 4500);			//2.4 ms (4500) is close - 0.6 ms (1125) is open - index
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_6, 1125);			//2.4 ms (4500) is close - 0.6 ms (1125) is open - thumb
	*/
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_1, 4500);			//0.6 ms (1125) is close - 2.4 ms (4500) is open - ring
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_2, 1125);			//0.6 ms (1125) is close - 2.4 ms (4500) is open - middle
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_3, 4500);			//2.4 ms (1125) is close - 0.6 ms (4500) is open - little
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_4, 4500);			//2.4 ms (4500) is close - 0.6 ms (1125) is open - index
	PWMPulseWidthSet(PWM0_BASE, PWM_OUT_6, 4500);			//2.4 ms (4500) is close - 0.6 ms (1125) is open - thumb
	//Enable PWM
	PWMGenEnable(PWM0_BASE, PWM_GEN_0);
	PWMGenEnable(PWM0_BASE, PWM_GEN_1);
	PWMGenEnable(PWM0_BASE, PWM_GEN_2);
	PWMGenEnable(PWM0_BASE, PWM_GEN_3);
	//Clear on compare
	PWMOutputState(PWM0_BASE, (PWM_OUT_1_BIT | PWM_OUT_2_BIT | PWM_OUT_3_BIT | PWM_OUT_4_BIT | PWM_OUT_6_BIT), true);
}

void Timers_Init(void){
	//Timers Configurations - Finger Speed Control
	SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER0); 										
	SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER2);										
	SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER3);										
	SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER4);										
	SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER5);			
	SysCtlPeripheralEnable(SYSCTL_PERIPH_TIMER6);										
	//Configure Timers as 32 bit
	TimerConfigure(TIMER0_BASE,TIMER_CFG_PERIODIC); 		
	TimerConfigure(TIMER2_BASE,TIMER_CFG_PERIODIC);									
	TimerConfigure(TIMER3_BASE,TIMER_CFG_PERIODIC);				
	TimerConfigure(TIMER4_BASE,TIMER_CFG_PERIODIC);									
	TimerConfigure(TIMER5_BASE,TIMER_CFG_PERIODIC);				
	TimerConfigure(TIMER6_BASE,TIMER_CFG_PERIODIC);									
	//Finger Speed
	TimerLoadSet(TIMER0_BASE, TIMER_A, 750000-1);				//thumb  servo motor - 1005 rad/seg - 160 Hz
	TimerLoadSet(TIMER2_BASE, TIMER_A, 750000-1);				//index  speed motor - 1005 rad/seg - 160 Hz
	TimerLoadSet(TIMER3_BASE, TIMER_A, 750000-1);				//middle speed motor - 1005 rad/seg - 160 Hz
	TimerLoadSet(TIMER4_BASE, TIMER_A, 750000-1);				//ring   speed motor - 1005 rad/seg - 160 Hz
	TimerLoadSet(TIMER5_BASE, TIMER_A, 750000-1);				//little speed motor - 1005 rad/seg - 160 Hz
	//ADC Timer
	TimerLoadSet(TIMER6_BASE, TIMER_A, 120000-1);				//ADC sample rate 1 kHz
	//Enable the specific interrupt vectors
	IntEnable(INT_TIMER0A);															
	IntEnable(INT_TIMER2A);																					
	IntEnable(INT_TIMER3A);																					
	IntEnable(INT_TIMER4A);																					
	IntEnable(INT_TIMER5A);																					
	IntEnable(INT_TIMER6A);																					
  //Overflow Interrupt
	TimerIntEnable(TIMER0_BASE, TIMER_TIMA_TIMEOUT);		
	TimerIntEnable(TIMER2_BASE, TIMER_TIMA_TIMEOUT);								
	TimerIntEnable(TIMER3_BASE, TIMER_TIMA_TIMEOUT);								
	TimerIntEnable(TIMER4_BASE, TIMER_TIMA_TIMEOUT);								
	TimerIntEnable(TIMER5_BASE, TIMER_TIMA_TIMEOUT);								
	TimerIntEnable(TIMER6_BASE, TIMER_TIMA_TIMEOUT);								
	//Enable the Timers
	TimerEnable(TIMER0_BASE, TIMER_A);									
	TimerEnable(TIMER2_BASE, TIMER_A);															
	TimerEnable(TIMER3_BASE, TIMER_A);															
	TimerEnable(TIMER4_BASE, TIMER_A);															
	TimerEnable(TIMER5_BASE, TIMER_A);															
	TimerEnable(TIMER6_BASE, TIMER_A);															
}

void ADC_Init(void){
	//ADC Initialization
	SysCtlPeripheralEnable(SYSCTL_PERIPH_ADC1);				
	//Hardware average	
	ADCHardwareOversampleConfigure(ADC1_BASE, 32);														
	//Sequence Disable	
	ADCSequenceDisable(ADC1_BASE, 2);												
	//ADC1, sample sequencer 2, high priority, trigger by software
	ADCSequenceConfigure(ADC1_BASE, 2, ADC_TRIGGER_PROCESSOR, 0);										   	
	//E1 is step 1, E2 is step 0, Ref is step 2
	ADCSequenceStepConfigure(ADC1_BASE, 2, 0, ADC_CTL_CH1);															//step 0 ch1
	ADCSequenceStepConfigure(ADC1_BASE, 2, 1, ADC_CTL_CH4);															//step 1 ch4
	ADCSequenceStepConfigure(ADC1_BASE, 2, 2, ADC_CTL_CH0 | ADC_CTL_IE | ADC_CTL_END);	//interrupt flag & end of sequence ch0
	//Sequence Enable	
	ADCSequenceEnable(ADC1_BASE, 2);	
}

void UART_init(void){
	//Enable PortA and UART0
	SysCtlPeripheralEnable(SYSCTL_PERIPH_GPIOA);
	SysCtlPeripheralEnable(SYSCTL_PERIPH_UART0);																				
	//Configure pins as UART
	GPIOPinConfigure(GPIO_PA0_U0RX);
  GPIOPinConfigure(GPIO_PA1_U0TX);
  GPIOPinTypeUART(GPIO_PORTA_BASE, GPIO_PIN_0 | GPIO_PIN_1);
	//UART Configuration
	UARTConfigSetExpClk(UART0_BASE, ui32SysClkFreq, 9600,(UART_CONFIG_WLEN_8 | UART_CONFIG_STOP_ONE | UART_CONFIG_PAR_NONE));
	IntEnable(INT_UART0);
	UARTIntEnable(UART0_BASE, UART_INT_RX | UART_INT_RT);
}

