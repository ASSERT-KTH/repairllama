from fl import fault_localizer, print_fault_localization

def test_simple_three_line_function():
    java_func = """public void sayHello() {
        System.out.println("Hello");
    }"""
    assert fault_localizer(java_func) == [(1, 1)]

def test_four_line_function():
    java_func = """public int add(int a, int b) {
        int result;
        result = a + b;
        return result;
    }"""
    expected = [
        (1, 1), (2, 2), (3, 3),  # Single line regions
        (1, 2), (2, 3), (1, 3)   # Multi-line regions
    ]
    assert sorted(fault_localizer(java_func)) == sorted(expected)

def test_six_line_function():
    java_func = """public String processString(String input) {
        if (input == null) {
            return "";
        }
        String result = input.trim();
        return result.toUpperCase();
    }"""
    expected = [
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),  # Single line regions
        (1, 2), (2, 3), (3, 4), (4, 5),          # Two line regions
        (1, 3), (2, 4), (3, 5),                  # Three line regions
        (1, 4), (2, 5),                          # Four line regions
        (1, 5)                                   # Five line regions
    ]
    assert sorted(fault_localizer(java_func)) == sorted(expected)

def test_eight_line_function():
    java_func = """public int factorial(int n) {
        if (n < 0) {
            throw new IllegalArgumentException();
        }
        int result = 1;
        for (int i = 1; i <= n; i++) {
            result *= i;
        }
        return result;
    }"""
    result = fault_localizer(java_func)
    
    # Calculate total number of regions for n lines
    n = 8  # number of lines in the function
    expected_regions = sum(n - i for i in range(n))  # or n * (n + 1) // 2
    
    assert len(result) == expected_regions
    assert all(start >= 1 and end <= 8 for start, end in result)

def test_single_statement_function():
    java_func = """public boolean isEmpty() {
        return size == 0;
    }"""
    assert fault_localizer(java_func) == [(1, 1)]  # Only one possible region

def test_print_fault_localization_four_line_function():
    java_func = """public int add(int a, int b) {
        int result;
        result = a + b;
        return result;
    }"""

    regions = fault_localizer(java_func)

    for region in regions:
        print(print_fault_localization(java_func, region))
