package com.samuelberrien.phyvr.normal;

import android.content.Context;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLSurfaceView;
import android.opengl.Matrix;
import android.util.AttributeSet;
import android.view.KeyEvent;
import android.view.MotionEvent;
import com.samuelberrien.phyvr.controls.Controls;
import com.samuelberrien.phyvr.wrappers.MainWrappers;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class MyGLSurfaceView extends GLSurfaceView implements GLSurfaceView.Renderer {

	private float[] projectionMatrix;
	private float[] viewMatrix;

	private Controls controls;

	private MainWrappers mainWrappers;

	public MyGLSurfaceView(Context context) {
		super(context);
		init();
	}

	public MyGLSurfaceView(Context context, AttributeSet attrs) {
		super(context, attrs);
		init();
	}

	private void init() {
		setEGLContextClientVersion(2);
		setRenderer(this);
		projectionMatrix = new float[16];
		viewMatrix = new float[16];
		Matrix.setIdentityM(viewMatrix, 0);
		mainWrappers = new MainWrappers(getContext());
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent motionEvent) {
		return controls.onMotionEvent(motionEvent) || super.onGenericMotionEvent(motionEvent);
	}

	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		return controls.onKeyDown(keyCode, event) || super.onKeyDown(keyCode, event);
	}

	@Override
	public void onSurfaceCreated(GL10 unused, EGLConfig eglConfig) {
		mainWrappers.init();
		controls = new Controls(getContext(), mainWrappers.getControlPtr());
	}

	@Override
	public void onSurfaceChanged(GL10 unused, int width, int height) {
		GLES20.glViewport(0, 0, width, height);

		float ratio = (float) width / height;

		Matrix.perspectiveM(projectionMatrix, 0, 40f, ratio, 0.1f, 500f);
	}

	@Override
	public void onDrawFrame(GL10 unused) {
		mainWrappers.update();
		mainWrappers.willDraw(viewMatrix, false);
		mainWrappers.draw(projectionMatrix, viewMatrix, new float[4], new float[3]);
	}

	public void free() {
		mainWrappers.free();
	}
}
