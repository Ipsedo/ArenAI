//
// Created by samuel on 26/03/2023.
//

#include <phyvr_utils/units.h>

float dp_to_px(AConfiguration *config, int dp) {
  int32_t density = AConfiguration_getDensity(config);

  /*switch (density_enum) {
    case ACONFIGURATION_DENSITY_LOW:
      density = 0.75f;
      break;
    case ACONFIGURATION_DENSITY_MEDIUM:
      density = 1.f;
      break;
    case ACONFIGURATION_DENSITY_HIGH:
      density = 1.5f;
      break;
    case ACONFIGURATION_DENSITY_XHIGH:
      density = 2.f;
      break;
    case ACONFIGURATION_DENSITY_XXHIGH:
      density = 3.f;
      break;
    case ACONFIGURATION_DENSITY_XXXHIGH:
      density = 4.f;
      break;
    default:
      density = 1.f;
  }*/

  return float(dp) * float(density) / 160.f;
}
