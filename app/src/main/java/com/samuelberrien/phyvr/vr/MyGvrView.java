package com.samuelberrien.phyvr.vr;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.AttributeSet;
import android.view.KeyEvent;
import android.view.MotionEvent;

import com.google.vr.sdk.base.Eye;
import com.google.vr.sdk.base.GvrView;
import com.google.vr.sdk.base.HeadTransform;
import com.google.vr.sdk.base.Viewport;
import com.samuelberrien.phyvr.controls.Controls;
import com.samuelberrien.phyvr.utils.LoadImage;
import com.samuelberrien.phyvr.wrappers.MainWrappers;

import javax.microedition.khronos.egl.EGLConfig;

public class MyGvrView extends GvrView implements GvrView.StereoRenderer {

	public MyGvrView(Context context) {
		super(context);
		mainWrappers = new MainWrappers(getContext(), true);
		setRenderer(this);
	}

	@Override
	public void onPause() {
		super.onPause();
	}

	@Override
	public void onResume() {
		super.onResume();
	}

	@Override
	public boolean onGenericMotionEvent(MotionEvent motionEvent) {
		return controls.onMotionEvent(motionEvent) || super.onGenericMotionEvent(motionEvent);
	}

	@Override
	public boolean onKeyDown(int keyCode, KeyEvent event) {
		return controls.onKeyDown(keyCode, event) || super.onKeyDown(keyCode, event);
	}

	/**
	 * Stereo Renderer stuff
	 */

	private final float Z_NEAR = 0.1f;
	private final float Z_FAR = 500f;

	private MainWrappers mainWrappers;

	private Controls controls;

	@Override
	public void onNewFrame(HeadTransform headTransform) {
		float[] mHeadView = new float[16];
		headTransform.getHeadView(mHeadView, 0);
		mainWrappers.willDraw(mHeadView, true);
		mainWrappers.update();
		//addBox(getContext().getAssets(), boxesPtr);
	}

	@Override
	public void onDrawEye(Eye eye) {
		float[] eyeView = eye.getEyeView();

		float[] mProjectionMatrix = eye.getPerspective(Z_NEAR, Z_FAR);
		mainWrappers.draw(mProjectionMatrix, eyeView, new float[4], new float[3]);
	}

	@Override
	public void onFinishFrame(Viewport viewport) {
	}

	@Override
	public void onSurfaceChanged(int width, int height) {
	}

	@Override
	public void onSurfaceCreated(EGLConfig config) {
		mainWrappers.init();
		controls = new Controls(getContext(), mainWrappers.getControlPtr());
	}

	@Override
	public void onRendererShutdown() {
		mainWrappers.free();
	}
}
