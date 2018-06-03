package com.samuelberrien.phyvr.normal;

import android.app.Activity;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.KeyEvent;
import android.view.MotionEvent;

public class PlayActivity extends Activity {

	private MyGLSurfaceView surfaceView;

	@Override
	protected void onCreate(@Nullable Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		surfaceView = new MyGLSurfaceView(this);
		setContentView(surfaceView);
	}

	@Override
	public void onPause() {
		surfaceView.onPause();
		super.onPause();
	}

	@Override
	public void onResume() {
		super.onResume();
		surfaceView.onResume();
	}

	@Override
	public boolean dispatchGenericMotionEvent(MotionEvent e) {
		return surfaceView.onGenericMotionEvent(e);
	}

	@Override
	public boolean dispatchKeyEvent(KeyEvent event) {
		surfaceView.onKeyDown(event.getKeyCode(), event);
		return event.getKeyCode() != KeyEvent.KEYCODE_BACK || super.dispatchKeyEvent(event);
	}
}
