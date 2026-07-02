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

public class Cursor extends View {

	private Paint paint;

	private int cursorColor;
	private int cursorColorAccent;
	private int containerColor;
	private int containerColorAccent;

	private float strokeWidth;
	private float cursorHeightRatio;

	private float cursorY;

	private boolean touched;
	private int pointerId;

	private boolean resetOnEnd;

	private CursorListener listener;

	public Cursor(Context context, AttributeSet attrs) {
		super(context, attrs);
		init(context, attrs);
	}

	public Cursor(Context context, AttributeSet attrs, int defStyleAttr) {
		super(context, attrs, defStyleAttr);
		init(context, attrs);
	}

	public Cursor(Context context, AttributeSet attrs, int defStyleAttr, int defStyleRes) {
		super(context, attrs, defStyleAttr, defStyleRes);
		init(context, attrs);
	}

	private void init(Context context, AttributeSet attrs) {
		TypedArray a = context.getTheme()
				.obtainStyledAttributes(attrs, R.styleable.Cursor, 0, 0);

		cursorColor = a.getColor(R.styleable.Cursor_cursor_color, ContextCompat.getColor(context, R.color.colorPrimary));
		cursorColorAccent = a.getColor(R.styleable.Cursor_cursor_color_accent, ContextCompat.getColor(context, R.color.colorAccent));
		containerColor = a.getColor(R.styleable.Cursor_container_color, ContextCompat.getColor(context, R.color.colorPrimary));
		containerColorAccent = a.getColor(R.styleable.Cursor_container_color_accent, ContextCompat.getColor(context, R.color.colorAccent));

		strokeWidth = a.getDimension(R.styleable.Cursor_cursor_stroke_width, 5.f);
		cursorHeightRatio = a.getFloat(R.styleable.Cursor_cursor_height_ratio, 0.2f);

		resetOnEnd = a.getBoolean(R.styleable.Cursor_reset_cursor_on_end, true);

		paint = new Paint();
		paint.setAntiAlias(true);
		paint.setStyle(Paint.Style.STROKE);
		paint.setStrokeWidth(strokeWidth);

		touched = false;
		pointerId = -1;
	}

	public void setListener(CursorListener listener) {
		this.listener = listener;
	}

	@Override
	protected void onLayout(boolean changed, int left, int top, int right, int bottom) {
		super.onLayout(changed, left, top, right, bottom);
		cursorY = getHeight() / 2.f;
	}

	@Override
	public boolean performClick() {
		return super.performClick();
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
				if (touched && currPointerId == pointerId) {
					pointerId = -1;
					touched = false;
					if (resetOnEnd) {
						cursorY = getHeight() / 2.f;

						if (listener != null) listener.onMove(0.f);

						invalidate();
					}
				}
				break;
			case MotionEvent.ACTION_MOVE:
				for (int i = 0; i < event.getPointerCount(); i++)
				if (touched && event.getPointerId(i) == pointerId) {
					float ptY = event.getY(i);
					float centeredY = ptY - getHeight() / 2.f;

					float maxDist = getHeight() / 2.f - getHeight() * cursorHeightRatio / 2.f - strokeWidth * 1.5f;

					if (Math.abs(centeredY) > maxDist) cursorY = centeredY * maxDist / Math.abs(centeredY) + getHeight() / 2.f;
					else cursorY = centeredY + getHeight() / 2.f;

					float relY = (cursorY - getHeight() / 2.f) / maxDist;

					if (listener != null) listener.onMove(relY);

					invalidate();
				}
				break;
		}

		return true;
	}

	@Override
	protected void onDraw(Canvas canvas) {
		if (touched) paint.setColor(containerColorAccent);
		else paint.setColor(containerColor);
		canvas.drawRect(0.f + strokeWidth / 2.f, 0.f + strokeWidth / 2.f, getWidth() - strokeWidth / 2.f, getHeight() - strokeWidth / 2.f, paint);

		if (touched) paint.setColor(cursorColorAccent);
		else paint.setColor(cursorColor);
		canvas.drawRect(0.f + strokeWidth * 1.5f,
				cursorY - getHeight() * cursorHeightRatio / 2.f,
				getWidth() - strokeWidth * 1.5f,
				cursorY + getHeight() * cursorHeightRatio / 2.f,
				paint);

	}

	public interface CursorListener {
		void onMove(float relY);
	}

	public void reset() {
		cursorY = getHeight() / 2.f;

		if (listener != null) listener.onMove(0.f);

		invalidate();
	}
}
