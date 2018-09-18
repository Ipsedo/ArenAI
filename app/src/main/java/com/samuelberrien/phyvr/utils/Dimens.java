package com.samuelberrien.phyvr.utils;

import android.content.Context;
import android.util.TypedValue;

public class Dimens {
	public static int dpResToPx(Context context, int dpResId) {
		float dp = context.getResources().getDimension(dpResId) / context.getResources().getDisplayMetrics().density;
		return (int) TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, dp,
				context.getResources().getDisplayMetrics());
	}

	public static final float Z_FAR = 2000f * (float) Math.sqrt(3);
	public static final float Z_NEAR = 0.1f;
}
