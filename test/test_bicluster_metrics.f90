program test_bicluster_metrics
  use sklearn_metrics, only: consensus_score
  use sklearn_common, only: dp,require
  use, intrinsic :: iso_fortran_env, only: int64
  use, intrinsic :: ieee_arithmetic, only: ieee_value,ieee_negative_inf
  implicit none
  logical :: rows(2,3),cols(2,3)
  real(dp) :: table(2,2),ids(2,1),actual
  rows(1,:)=[.true.,.true.,.false.];rows(2,:)=[.false.,.false.,.true.]
  cols(1,:)=[.false.,.true.,.true.];cols(2,:)=[.true.,.false.,.false.]
  actual=consensus_score(rows,cols,rows(2:1:-1,:),cols(2:1:-1,:))
  call require(abs(actual-1)<1.e-14_dp,'Bicluster consensus is invariant to cluster ordering')
  actual=consensus_score(merge(1,0,rows),merge(1_int64,0_int64,cols),rows,cols)
  call require(abs(actual-1)<1.e-14_dp,'Bicluster consensus mixed logical and integer types')
  actual=consensus_score(rows,cols,rows(:1,:),cols(:1,:))
  call require(abs(actual-.5_dp)<1.e-14_dp,'Bicluster consensus normalizes by the larger set')
  call require(consensus_score(rows(:1,:),cols(:1,:),rows(2:,:),cols(2:,:))==0, &
    'Disjoint biclusters have zero consensus')
  ids(:,1)=[1._dp,2._dp];table(1,:)=[.9_dp,.8_dp];table(2,:)=[.85_dp,.1_dp]
  actual=consensus_score(ids,ids,ids,ids,similarity=similarity)
  call require(abs(actual-.825_dp)<1.e-14_dp,'Assignment finds a non-greedy optimum')
  table=ieee_value(0._dp,ieee_negative_inf);table(1,1)=1;table(2,2)=2
  actual=consensus_score(ids,ids,ids,ids,similarity=similarity)
  call require(abs(actual-1.5_dp)<1.e-14_dp,'Assignment avoids forbidden negative-infinity similarities')
  print *, 'Bicluster consensus, rectangular optimal matching and callbacks: PASS'
contains
  real(dp) function similarity(ar,ac,br,bc) result(value)
    real(dp), intent(in) :: ar(:),ac(:),br(:),bc(:)
    call require(size(ac)>0.and.size(bc)>0,'Callback receives column indicators')
    value=table(nint(ar(1)),nint(br(1)))
  end function
end program
