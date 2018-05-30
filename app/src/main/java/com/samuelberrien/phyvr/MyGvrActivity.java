package com.samuelberrien.phyvr;

import android.os.Bundle;

import com.google.vr.sdk.base.GvrActivity;

public class MyGvrActivity extends GvrActivity {

	private MyGvrView myGvrView;

	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);

		myGvrView = new MyGvrView(this);
		setContentView(myGvrView);
	}

	@Override
	protected void onResume() {
		super.onResume();
		myGvrView.onResume();
	}

	@Override
	protected void onPause() {
		myGvrView.onPause();
		super.onPause();
	}
}
