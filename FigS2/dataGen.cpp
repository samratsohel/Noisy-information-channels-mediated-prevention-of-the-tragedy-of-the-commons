#include <main.h>

int main()
{
int scheme = 1; // states' perception channel simultaneous moves
/////////////////////////////////////////////////////////////////////////////////////



	////////////////////  All data egenration for Fig. 2. ///////////////////////////
	// trajDataParamsSave(scheme);

	////////////////////  All data egenration for Fig. 3. ///////////////////////////
	// spaceDataParamsSave(scheme);

	////////////////////  All data egenration for Fig. 4. ///////////////////////////
	// AspaceDataParamsSave(scheme);

	////////////////////  All data egenration for Fig. 5. //////////////////////////
	// MarkovChain(scheme);

	////////////////////  All data egenration for Fig. 6. //////////////////////////
	// mechanism(scheme);

	//////////////////  All data egenration for Fig. S1. /////////////////////////
	// disspaceDataParamsSave(scheme);



scheme = 2; // states' perception channel alternating moves
/////////////////////////////////////////////////////////////////////////////////////



	////////////////////  All data egenration for Fig. S2. /////////////////////////
	// trajDataParamsSave(scheme);

	////////////////////  All data egenration for Fig. S3. /////////////////////////
	// spaceDataParamsSave(scheme);

	////////////////////  All data egenration for Fig. S4. /////////////////////////
	// AspaceDataParamsSave(scheme);

	////////////////////  All data egenration for Fig. S5. /////////////////////////
	// disspaceDataParamsSave(scheme);



scheme = 1; // actions' perception channel simulataneous
/////////////////////////////////////////////////////////////////////////////////////



	////////////////////  All data egenration for Fig. S6. /////////////////////////
	// trajDataParamsSaveW(scheme);

	////////////////////  All data egenration for Fig. S7. /////////////////////////
	// spaceDataParamsSaveW(scheme);

	////////////////////  All data egenration for Fig. S8. /////////////////////////
	// AspaceDataParamsSaveW(scheme);

	////////////////////  All data egenration for Fig. S9. /////////////////////////
	// disspaceDataParamsSaveW(scheme);



scheme = 2; // actions' perception channel alternating
/////////////////////////////////////////////////////////////////////////////////////



	////////////////////  All data egenration for Fig. S10. /////////////////////////
	// trajDataParamsSaveW(scheme);

	////////////////////  All data egenration for Fig. S11. /////////////////////////
	// spaceDataParamsSaveW(scheme);

	////////////////////  All data egenration for Fig. S12. /////////////////////////
	// AspaceDataParamsSaveW(scheme);

	////////////////////  All data egenration for Fig. S13. /////////////////////////
	// disspaceDataParamsSaveW(scheme);



scheme = 1; // both the channels simulatenous
/////////////////////////////////////////////////////////////////////////////////////


	
	////////////////////  All data egenration for Fig. S14. /////////////////////////
	// AspaceDataParamsSaveNW(scheme);



scheme = 2; // both the channels alternating
/////////////////////////////////////////////////////////////////////////////////////



	////////////////////  All data egenration for Fig. S15. /////////////////////////
	// AspaceDataParamsSaveNW(scheme);

//Convergence test
scheme = 1;
/////////////////////////////////////////////////////////////////////////////////////
	spaceDataParamsSaveCT(scheme);
	return 0;
}