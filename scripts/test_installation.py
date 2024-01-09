from fibsem.microscope import _THERMO_API_AVAILABLE, _TESCAN_API_AVAILABLE



def main():

    # OpenFIBSEM API
    print(f"\n\nOpenFIBSEM API:\n")
    try:
        import fibsem
        FIBSEM_AVAILABLE = True
    except ImportError:
        FIBSEM_AVAILABLE = False
    if FIBSEM_AVAILABLE:
        print(f"OpenFIBSEM v{fibsem.__version__}")
        print(f"Installed at: {fibsem.__path__}")
    
    print(f"-" * 80)
    
    print(f"Applications:\n")
    try: 
        import autolamella 
        AUTOLAMELLA_AVAILABLE = True
    except ImportError: 
        AUTOLAMELLA_AVAILABLE = False
    if AUTOLAMELLA_AVAILABLE:
        print(f"AutoLamella v{autolamella.__version__}")
        print(f"Installed at: {autolamella.__path__}")   
    
    try: 
        import salami
        SALAMI_AVAILABLE = True
    except ImportError:
        SALAMI_AVAILABLE = False
    if SALAMI_AVAILABLE:
        print(f"SALAMI v{salami.__version__}")
        print(f"Installed at: {salami.__path__}")
    print(f"-" * 80)
    
    # Hardware APIs
    print(f"Hardware APIs:\n")
    
    # Thermo Fisher API
    print(f"ThermoFisher API {'Available' if _THERMO_API_AVAILABLE else 'Not Available'}")
    if _THERMO_API_AVAILABLE:
        from fibsem.microscope import version as autoscript_version
        print(f"AutoScript v{autoscript_version}")
    print(f"-" * 80)

    # Tescan API
    print(f"Tescan API {'Available' if _TESCAN_API_AVAILABLE else 'Not Available'}")
    if _TESCAN_API_AVAILABLE:
        from fibsem.microscope import tescanautomation
        print(f"TescanAutomation v{tescanautomation.__version__}")

if __name__ == "__main__":
    main()