package com.samuelberrien.arenai.set_controls.ui;

import android.content.Context;
import android.content.res.TypedArray;
import android.graphics.Canvas;
import android.graphics.Paint;
import androidx.core.content.ContextCompat;
import android.util.AttributeSet;
import android.view.MotionEvent;
import android.view.View;

import com.samuelberrien.arenai.R;

public class JoyStick extends View {

	private boolean resetStickOnEnd;

	private float stickRadiusRatio;
	private float strokeWidth;

	private Paint paint;

	private int pointerId;
	private boolean touched;

	private float joystickCenterX;
	private float joystickCenterY;

	private JoyStickListener listener;

	private int stickColor;
	private int stickColorAccent;

	private int circleColor;
	private int circleColorAccent;

	public JoyStick(Context context, AttributeSet attrs) {
		super(context, attrs);
		init(context, attrs);
	}

	public JoyStick(Context context, AttributeSet attrs, int defStyleAttr) {
		super(context, attrs, defStyleAttr);
		init(context, attrs);
	}

	public JoyStick(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
		super(context, attrs, defStyleAttr, defStyleRes);
		init(context, attrs);
	}

	private void init(Context context, AttributeSet attrs) {
		TypedArray a = context.getTheme()
				.obtainStyledAttributes(attrs, R.styleable.JoyStick, 0, 0);

		resetStickOnEnd = a.getBoolean(R.styleable.JoyStick_reset_stick_on_end, true);

		stickRadiusRatio = a.getFloat(R.styleable.JoyStick_stick_radius_ratio, 0.3f);
		strokeWidth = a.getDimension(R.styleable.JoyStick_joystick_stroke_width, 5.f);

		stickColor = a.getColor(R.styleable.JoyStick_stick_color, ContextCompat.getColor(context, R.color.colorPrimary));
		stickColorAccent = a.getColor(R.styleable.JoyStick_stick_color_accent, ContextCompat.getColor(context, R.color.colorAccent));
		circleColor = a.getColor(R.styleable.JoyStick_circle_color, ContextCompat.getColor(context, R.color.colorPrimary));
		circleColorAccent = a.getColor(R.styleable.JoyStick_circle_color_accent, ContextCompat.getColor(context, R.color.colorAccent));

		listener = null;

		pointerId = -1;
		touched = false;

		paint = new Paint();
		paint.setAntiAlias(true);
		paint.setStyle(Paint.Style.STROKE);
		paint.setStrokeWidth(strokeWidth);
	}

	public void setJoyStickListener(JoyStickListener listener) {
		this.listener = listener;
	}

	@Override
	protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
		super.onLayout(changed, left, top, right, bottom);
		joystickCenterX = getWidth() / 2.f;
		joystickCenterY = getWidth() / 2.f;
	}

	@Override
	protected void onDraw(Canvas canvas) {
		if (touched) paint.setColor(circleColorAccent);
		else paint.setColor(circleColor);
		canvas.drawCircle(getWidth() / 2.f, getWidth() / 2.f, getWidth() / 2.f - strokeWidth / 2.f, paint);

		if (touched) paint.setColor(stickColorAccent);
		else paint.setColor(stickColor);
		canvas.drawCircle(joystickCenterX, joystickCenterY, stickRadiusRatio * getWidth() / 2.f, paint);
	}

	@Override
	public void onDrawForeground(Canvas canvas) {
		super.onDrawForeground(canvas);
	}

	@Override
	public boolean onTouchEvent(MotionEvent event) {
		int currPointerId = event.getPointerId(event.getActionIndex());

		if (touched && currPointerId != pointerId) return false;

		switch (event.getAction()) {
			case MotionEvent.ACTION_DOWN:
				if (!touched) {
					touched = true;
					pointerId = currPointerId;

					performClick();
				}
				break;
			case MotionEvent.ACTION_UP:
				if (touched && pointerId == currPointerId) {
					touched = false;
					pointerId = -1;

					if (resetStickOnEnd) {
						joystickCenterX = getWidth() / 2.f;
						joystickCenterY = getWidth() / 2.f;

						if (listener != null) listener.onMove(0.f, 0.f);
					}

					invalidate();
				}
				break;
			case MotionEvent.ACTION_MOVE:

				for (int i = 0; i < event.getPointerCount(); i++)
				if (touched && event.getPointerId(i) == pointerId) {
					float ptX = event.getX(i);
					float ptY = event.getY(i);
					float centeredX = ptX - getWidth() / 2.f;
					float centeredY = ptY - getHeight() / 2.f;

					float dist = (float) Math.sqrt(Math.pow(centeredX, 2.) + Math.pow(centeredY, 2.));

					float maxDist = getWidth() / 2.f - strokeWidth * 1.5f - stickRadiusRatio * getWidth() / 2.f;

					if (dist > maxDist) {
						joystickCenterX = maxDist * centeredX / dist + getWidth() / 2.f;
						joystickCenterY = maxDist * centeredY / dist + getHeight() / 2.f;
					} else {
						joystickCenterX = ptX;
						joystickCenterY = ptY;
					}

					float relX = (joystickCenterX - getWidth() / 2.f) / maxDist;
					float relY = (joystickCenterY - getHeight() / 2.f) / maxDist;

					if (listener != null) listener.onMove(relX, relY);

					invalidate();
				}
				break;
		}
		return true;
	}

	@Override
	public boolean performClick() {
		return super.performClick();
	}

	@Override
	public void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
		int width = MeasureSpec.getSize(widthMeasureSpec);
		int height = MeasureSpec.getSize(heightMeasureSpec);
		int size = width > height ? height : width;
		setMeasuredDimension(size, size);
	}

	public interface JoyStickListener {

		void onMove(float relX, float relY);

	}
}
