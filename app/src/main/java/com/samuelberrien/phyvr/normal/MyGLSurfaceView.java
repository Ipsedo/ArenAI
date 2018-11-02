package com.samuelberrien.phyvr.normal;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.AttributeSet;
import android.view.KeyEvent;
import android.view.MotionEvent;
import com.samuelberrien.phyvr.MainActivity;
import com.samuelberrien.phyvr.controls.Controls;
import com.samuelberrien.phyvr.wrappers.MainWrappers;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.opengles.GL10;

import static com.samuelberrien.phyvr.utils.Dimens.Z_FAR;
import static com.samuelberrien.phyvr.utils.Dimens.Z_NEAR;

public class MyGLSurfaceView extends GLSurfaceView implements GLSurfaceView.Renderer, GLSurfaceView.EGLConfigChooser {

	private float[] projectionMatrix;
	private float[] viewMatrix;

	private Controls controls;

	private MainWrappers mainWrappers;

	private boolean willQuit;

	public MyGLSurfaceView(Context context) {
		super(context);
		init();
	}

	public MyGLSurfaceView(Context context, AttributeSet attrs) {
		super(context, attrs);
		init();
	}

	private void init() {
		willQuit = false;
		setEGLConfigChooser(this);
		setEGLContextClientVersion(3);
		setPreserveEGLContextOnPause(true);
		setRenderer(this);
		projectionMatrix = new float[16];
		viewMatrix = new float[16];
		Matrix.setIdentityM(viewMatrix, 0);
		initWrappers();
	}

	private void initWrappers() {
		mainWrappers = new MainWrappers(getContext(),
				false,
				((Activity) getContext()).getIntent().getIntExtra(MainActivity.levelIdxExtraStr, 0));
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent motionEvent) {
		if (mainWrappers == null || !mainWrappers.isInit())
			return super.onGenericMotionEvent(motionEvent);
		return controls.onMotionEvent(motionEvent) || super.onGenericMotionEvent(motionEvent);
	}

	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		if (mainWrappers == null || !mainWrappers.isInit())
			return super.onKeyDown(keyCode, event);
		return controls.onKeyDown(keyCode, event) || super.onKeyDown(keyCode, event);
	}

	@Override
	public boolean onKeyUp(int keyCode, KeyEvent event) {
		if (mainWrappers == null || !mainWrappers.isInit())
			return super.onKeyUp(keyCode, event);
		return controls.onKeyUp(keyCode, event) || super.onKeyUp(keyCode, event);
	}

	@Override
	public void onSurfaceCreated(GL10 unused, EGLConfig eglConfig) {
		if (!mainWrappers.isInit()) {
			mainWrappers.init();
			controls = new Controls(getContext(), mainWrappers.getLevelPtr());
		}
	}

	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height) {
		GLES20.glViewport(0, 0, width, height);

		float ratio = (float) width / height;

		Matrix.perspectiveM(projectionMatrix, 0, 40f, ratio, Z_NEAR, Z_FAR);
	}

	@Override
	public void onDrawFrame(GL10 unused) {
		controls.sendInputs();
		mainWrappers.update();
		mainWrappers.willDraw(viewMatrix, false);
		mainWrappers.draw(projectionMatrix, viewMatrix, new float[3], new float[3]);
		detectWinLose();
	}

	private void detectWinLose() {
		if(willQuit)
			return;

		boolean lose = mainWrappers.lose();

		if (lose || mainWrappers.win()) {
			willQuit = true;
			post(() -> {
				new AlertDialog.Builder(getContext())
						.setMessage(lose ? "Game Over !" : "Game Done !")
						.setNeutralButton("Main menu",
								(DialogInterface dialogInterface, int i) -> {
									((Activity) getContext()).finish();
								})
						.setCancelable(false)
						.create()
						.show();
				setRenderMode(RENDERMODE_WHEN_DIRTY);
			});
		}
	}

	public void free() {
		mainWrappers.free();
	}

	@Override
	public EGLConfig chooseConfig(EGL10 egl10, EGLDisplay eglDisplay) {
		int attribs[] = {
				EGL10.EGL_LEVEL, 0,
				EGL10.EGL_RENDERABLE_TYPE, 4,
				EGL10.EGL_COLOR_BUFFER_TYPE, EGL10.EGL_RGB_BUFFER,
				EGL10.EGL_RED_SIZE, 8,
				EGL10.EGL_GREEN_SIZE, 8,
				EGL10.EGL_BLUE_SIZE, 8,
				EGL10.EGL_DEPTH_SIZE, 16,
				EGL10.EGL_SAMPLE_BUFFERS, 1,
				EGL10.EGL_SAMPLES, 4,
				EGL10.EGL_NONE
		};
		EGLConfig[] configs = new EGLConfig[1];
		int[] configCounts = new int[1];
		egl10.eglChooseConfig(eglDisplay, attribs, configs, 1, configCounts);

		if (configCounts[0] == 0) {
			return null;
		} else {
			return configs[0];
		}
	}
}
