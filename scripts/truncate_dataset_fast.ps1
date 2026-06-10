# Fast dataset truncation using PowerShell batch operations
param(
    [int]$MaxImagesPerClass = 4000
)

$datasetRoot = "$(Split-Path -Parent $PSScriptRoot)\dataset"
$partitions = @("train", "val", "test")

function CountImagesPerClass {
    $classCounts = @{}
    
    foreach ($partition in $partitions) {
        $partitionPath = Join-Path $datasetRoot $partition
        if (Test-Path $partitionPath) {
            Get-ChildItem -Path $partitionPath -Directory | ForEach-Object {
                $className = $_.Name
                $imageCount = (Get-ChildItem -Path $_.FullName -File -Recurse | Measure-Object).Count
                
                if (-not $classCounts[$className]) {
                    $classCounts[$className] = @{}
                }
                $classCounts[$className][$partition] = $imageCount
            }
        }
    }
    
    return $classCounts
}

function TruncateClassFast {
    param(
        [string]$ClassName,
        [hashtable]$PartitionCounts
    )
    
    $totalImages = 0
    $partitionCounts.Values | ForEach-Object { $totalImages += $_ }
    
    if ($totalImages -le $MaxImagesPerClass) {
        Write-Host "  OK $ClassName`: $totalImages images (no truncation needed)"
        return
    }
    
    # Calculate target counts per partition
    $reductionRatio = $MaxImagesPerClass / $totalImages
    $targetCounts = @{}
    
    foreach ($partition in $partitions) {
        $target = [Math]::Max(1, [int]($partitionCounts[$partition] * $reductionRatio))
        $targetCounts[$partition] = $target
    }
    
    # Adjust to hit exactly MAX_IMAGES_PER_CLASS
    $currentTotal = 0
    $targetCounts.Values | ForEach-Object { $currentTotal += $_ }
    
    if ($currentTotal -lt $MaxImagesPerClass) {
        $diff = $MaxImagesPerClass - $currentTotal
        $largestPartition = ($partitionCounts.GetEnumerator() | Sort-Object -Property Value -Descending)[0].Name
        $targetCounts[$largestPartition] += $diff
    }
    elseif ($currentTotal -gt $MaxImagesPerClass) {
        $diff = $currentTotal - $MaxImagesPerClass
        foreach ($partition in @("train", "val", "test") | Sort-Object { $partitionCounts[$_] } -Descending) {
            if ($diff -le 0) { break }
            $reduction = [Math]::Min($diff, ($targetCounts[$partition] - 1))
            if ($reduction -gt 0) {
                $targetCounts[$partition] -= $reduction
                $diff -= $reduction
            }
        }
    }
    
    $newTotal = ($targetCounts.Values | Measure-Object -Sum).Sum
    Write-Host "  Truncating $ClassName`: $totalImages -> $newTotal images"
    
    # Truncate each partition
    foreach ($partition in $partitions) {
        $partitionPath = Join-Path $datasetRoot $partition $ClassName
        if (-not (Test-Path $partitionPath)) { continue }
        
        $currentCount = $partitionCounts[$partition]
        $targetCount = $targetCounts[$partition]
        
        if ($currentCount -le $targetCount) {
            Write-Host "    $partition`: $currentCount -> $targetCount (no change)"
            continue
        }
        
        # Get all images
        $allImages = Get-ChildItem -Path $partitionPath -File -Recurse
        $imagesToKeep = $targetCount
        $imagesToDelete = $currentCount - $targetCount
        
        # Shuffle and select images to delete
        $imagesToDelete = $allImages | Get-Random -Count $imagesToDelete
        
        # Delete in batches for faster processing
        $batchSize = 100
        $deletedCount = 0
        
        for ($i = 0; $i -lt $imagesToDelete.Count; $i += $batchSize) {
            $batch = $imagesToDelete[$i..([Math]::Min($i + $batchSize - 1, $imagesToDelete.Count - 1))]
            $batch | Remove-Item -Force -ErrorAction SilentlyContinue
            $deletedCount += $batch.Count
        }
        
        Write-Host "    $partition`: $currentCount -> $targetCount (deleted $($imagesToDelete.Count))"
    }
}

# Main execution
Write-Host "================================================================"
Write-Host "FAST DATASET TRUNCATION SCRIPT (PowerShell)"
Write-Host "================================================================"
Write-Host "Max images per class: $MaxImagesPerClass"
Write-Host ""

Write-Host "Counting current images per class..."
$classCounts = CountImagesPerClass

# Find classes exceeding limit
$classesToTruncate = @()
$classCounts.GetEnumerator() | ForEach-Object {
    $className = $_.Name
    $partitionCounts = $_.Value
    $total = 0
    $partitionCounts.Values | ForEach-Object { $total += $_ }
    
    if ($total -gt $MaxImagesPerClass) {
        $classesToTruncate += @{ Name = $className; Counts = $partitionCounts; Total = $total }
    }
}

if ($classesToTruncate.Count -eq 0) {
    Write-Host "✓ No classes exceed the limit. Dataset is already balanced."
    return
}

Write-Host "Found $($classesToTruncate.Count) classes to truncate:"
Write-Host ""
$classesToTruncate | ForEach-Object {
    Write-Host "  • $($_.Name): $($_.Total) images"
}

Write-Host ""
Write-Host "================================================================"
$response = Read-Host "Proceed with truncation? (yes/no)"
if ($response -ne "yes") {
    Write-Host "Truncation cancelled."
    return
}

Write-Host ""
Write-Host "Truncating classes..."
Write-Host "================================================================"

$totalDeleted = 0
$classesToTruncate | ForEach-Object {
    $beforeTotal = $_.Total
    TruncateClassFast -ClassName $_.Name -PartitionCounts $_.Counts
    $totalDeleted += $beforeTotal - $MaxImagesPerClass
}

# Verify results
Write-Host ""
Write-Host "================================================================"
Write-Host "Verifying truncation..."
$updatedCounts = CountImagesPerClass

$maxInClass = 0
$updatedCounts.Values | ForEach-Object {
    $total = 0
    $_.Values | ForEach-Object { $total += $_ }
    if ($total -gt $maxInClass) { $maxInClass = $total }
}

Write-Host "Max images per class after truncation: $maxInClass"
Write-Host "Total images deleted: $totalDeleted"
Write-Host ""
Write-Host "OK - Truncation complete!"

# Summary statistics
Write-Host ""
Write-Host "================================================================"
Write-Host "SUMMARY STATISTICS"
Write-Host "================================================================"
Write-Host ""

$partitionTotals = @{}
foreach ($partition in $partitions) {
    $partitionTotals[$partition] = 0
}

$updatedCounts.Values | ForEach-Object {
    $_.GetEnumerator() | ForEach-Object {
        $partitionTotals[$_.Name] += $_.Value
    }
}

$totalImages = 0
$partitionTotals.Values | ForEach-Object { $totalImages += $_ }

Write-Host "Total images across dataset: $totalImages"
foreach ($partition in $partitions) {
    $pct = if ($totalImages -gt 0) { ($partitionTotals[$partition] / $totalImages * 100) } else { 0 }
    $pctRounded = [Math]::Round($pct, 1)
    $pctStr = $pctRounded.ToString() + " percent"
    Write-Host ("  " + $partition + ": " + $partitionTotals[$partition] + " (" + $pctStr + ")")
}

Write-Host ""
Write-Host ("Total classes: " + $updatedCounts.Count)
