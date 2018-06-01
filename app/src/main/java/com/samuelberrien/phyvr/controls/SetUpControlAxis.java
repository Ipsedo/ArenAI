package com.samuelberrien.phyvr.controls;

import android.app.Activity;

public class SetUpControlAxis extends SetUpControl {
	public SetUpControlAxis(Activity activity, int controlStrId) {
		super(activity, controlStrId);
		infos.isMotionEvent = true;
	}
}
