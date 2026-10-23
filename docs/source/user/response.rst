LISA Response Models
~~~~~~~~~~~~~~~~~~~~~~~~~~

Fast Frequency Domain Response
***********************************

:class:`~bbhx.response.fastfdresponse.LISATDIResponse` computes the transfer functions of
each harmonic along its time-frequency track t(f). It reads the spacecraft positions from
a LISA Analysis Tools orbit object; pass that object explicitly, with the same backend
flavor. The sky angles must be in the frame of the orbit data (see :doc:`conventions`).
The time-domain response used by :class:`~bbhx.mbhtdionfly.MBHTDIonFly` and
:class:`~bbhx.sobbhtdionfly.SOBBHTDIonFly` lives in LISA Analysis Tools
(:class:`lisatools.response.tdionfly.TDTDIonTheFly`).

.. autoclass:: bbhx.response.fastfdresponse.LISATDIResponse
    :members:
    :show-inheritance:
    :inherited-members:
