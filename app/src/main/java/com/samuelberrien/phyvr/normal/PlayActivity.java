package com.samuelberrien.phyvr.normal;

import android.app.Activity;
import android.os.Bundle;
import android.support.annotation.Nullable;
import android.view.KeyEvent;
import android.view.MotionEvent;
import android.view.WindowManager;

import com.samuelberrien.phyvr.MainActivity;
import com.samuelberrien.phyvr.R;

public class PlayActivity extends Activity {

	private MyGLSurfaceView surfaceView;

	@Override
	protected void onCreate(@Nullable Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		getWindow().setFlags(
				WindowManager.LayoutParams.FLAG_FULLSCREEN,
				WindowManager.LayoutParams.FLAG_FULLSCREEN);
		setContentView(R.layout.normal_game);
		surfaceView = findViewById(R.id.surface_view);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
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
		boolean handled = false;
		if (event.getAction() == KeyEvent.ACTION_DOWN)
			handled = surfaceView.onKeyDown(event.getKeyCode(), event);
		else if (event.getAction() == KeyEvent.ACTION_UP)
			handled = surfaceView.onKeyUp(event.getKeyCode(), event);
		return handled
				|| event.getKeyCode() != KeyEvent.KEYCODE_BACK
				|| super.dispatchKeyEvent(event);
	}

	@Override
	protected void onDestroy() {
		surfaceView.free();
		super.onDestroy();
	}
}
