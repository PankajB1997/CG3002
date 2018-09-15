#include <stdlib.h>

void setup()
{
  // put your setup code here, to run once:
  Serial.begin(115200);
  Serial1.begin(115200);
  handshake();
  
}


void handshake()
{
  int h_flag = 0;
  int n_flag = 0;


  while (h_flag == 0){
    if (Serial1.available()) {
      if ((Serial1.read() == 'H')) {
            h_flag = 1;
          }
    }
  }

  while (n_flag ==0 ) {
    if (Serial1.available()) {
        Serial1.write('A');
        if (Serial1.read() == 'N'){
          Serial.println("connected via N");
         n_flag = 1;
        }
    }
  }      

}
void loop(){}

