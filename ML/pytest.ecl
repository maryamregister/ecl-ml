//First form: a structure
IMPORT * FROM ML;
IMPORT STD;
IMPORT * FROM $;
IMPORT plugins;
IMPORT Python;  //make Python language available

INTEGER addone(INTEGER p) := EMBED(Python)
  return 2
ENDEMBED;

output (addone(2));
